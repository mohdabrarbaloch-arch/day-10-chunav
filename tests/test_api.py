"""Tests for the HTTP layer: auth, datasets, runs, predictions."""

from __future__ import annotations


class TestHealth:
    def test_health_is_public(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Chunav" in r.text


class TestAuth:
    def test_register_returns_a_token(self, client):
        r = client.post(
            "/api/auth/register",
            json={"email": "new@test.pk", "display_name": "New", "password": "password-123"},
        )
        assert r.status_code == 201
        assert r.json()["access_token"]

    def test_duplicate_email_is_409(self, client):
        body = {"email": "dup@test.pk", "display_name": "D", "password": "password-123"}
        assert client.post("/api/auth/register", json=body).status_code == 201
        assert client.post("/api/auth/register", json=body).status_code == 409

    def test_short_password_is_422(self, client):
        r = client.post(
            "/api/auth/register",
            json={"email": "s@test.pk", "display_name": "S", "password": "short"},
        )
        assert r.status_code == 422

    def test_login_works(self, client):
        client.post(
            "/api/auth/register",
            json={"email": "li@test.pk", "display_name": "L", "password": "password-123"},
        )
        r = client.post("/api/auth/login", json={"email": "li@test.pk", "password": "password-123"})
        assert r.status_code == 200

    def test_wrong_password_is_401(self, client):
        client.post(
            "/api/auth/register",
            json={"email": "wp@test.pk", "display_name": "W", "password": "password-123"},
        )
        r = client.post("/api/auth/login", json={"email": "wp@test.pk", "password": "wrong-pass"})
        assert r.status_code == 401

    def test_unknown_user_gives_the_same_message(self, client):
        # Identical detail for both failure modes — no user enumeration.
        r = client.post("/api/auth/login", json={"email": "ghost@test.pk", "password": "x"})
        assert r.status_code == 401
        assert r.json()["detail"] == "Incorrect email or password"

    def test_me_requires_a_token(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_me_returns_the_user(self, client, auth):
        headers, user = auth
        r = client.get("/api/auth/me", headers=headers)
        assert r.status_code == 200
        assert r.json()["email"] == user["email"]

    def test_garbage_token_is_401(self, client):
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
        assert r.status_code == 401


class TestDatasets:
    def test_profile_does_not_persist(self, client, auth, churn_csv):
        headers, _ = auth
        r = client.post("/api/datasets/profile", json={"csv_text": churn_csv}, headers=headers)
        assert r.status_code == 200
        assert r.json()["n_rows"] == 400
        assert client.get("/api/datasets", headers=headers).json() == []

    def test_profile_suggests_a_target(self, client, auth, churn_csv):
        headers, _ = auth
        r = client.post("/api/datasets/profile", json={"csv_text": churn_csv}, headers=headers)
        assert r.json()["suggested_target"] == "churned"

    def test_create_dataset(self, client, auth, churn_csv):
        headers, _ = auth
        r = client.post(
            "/api/datasets",
            json={"name": "Churn", "target_column": "churned", "csv_text": churn_csv},
            headers=headers,
        )
        assert r.status_code == 201
        assert r.json()["task"] == "classification"

    def test_duplicate_name_is_409(self, client, auth, churn_csv):
        headers, _ = auth
        body = {"name": "Same", "target_column": "churned", "csv_text": churn_csv}
        assert client.post("/api/datasets", json=body, headers=headers).status_code == 201
        assert client.post("/api/datasets", json=body, headers=headers).status_code == 409

    def test_bad_target_is_422(self, client, auth, churn_csv):
        headers, _ = auth
        r = client.post(
            "/api/datasets",
            json={"name": "Bad", "target_column": "nope", "csv_text": churn_csv},
            headers=headers,
        )
        assert r.status_code == 422

    def test_datasets_are_private(self, client, auth, churn_csv):
        headers_a, _ = auth
        created = client.post(
            "/api/datasets",
            json={"name": "Mine", "target_column": "churned", "csv_text": churn_csv},
            headers=headers_a,
        ).json()

        # A second user must not see or fetch it.
        r = client.post(
            "/api/auth/register",
            json={"email": "other@test.pk", "display_name": "O", "password": "password-123"},
        )
        other = {"Authorization": f"Bearer {r.json()['access_token']}"}
        assert client.get("/api/datasets", headers=other).json() == []
        assert client.get(f"/api/datasets/{created['id']}", headers=other).status_code == 404

    def test_delete_removes_it(self, client, auth, churn_csv):
        headers, _ = auth
        created = client.post(
            "/api/datasets",
            json={"name": "Temp", "target_column": "churned", "csv_text": churn_csv},
            headers=headers,
        ).json()
        assert client.delete(f"/api/datasets/{created['id']}", headers=headers).status_code == 204
        assert client.get(f"/api/datasets/{created['id']}", headers=headers).status_code == 404


class TestRuns:
    def _dataset(self, client, headers, csv_text, name="D"):
        return client.post(
            "/api/datasets",
            json={"name": name, "target_column": "churned", "csv_text": csv_text},
            headers=headers,
        ).json()

    def test_run_returns_a_leaderboard(self, client, auth, churn_csv):
        headers, _ = auth
        ds = self._dataset(client, headers, churn_csv)
        r = client.post(
            "/api/runs",
            json={"dataset_id": ds["id"], "n_folds": 3, "max_candidates": 3},
            headers=headers,
        )
        assert r.status_code == 201
        body = r.json()
        assert body["winner_label"]
        assert len(body["leaderboard"]) >= 1
        assert body["leaderboard"][0]["rank"] == 1

    def test_run_on_someone_elses_dataset_is_404(self, client, auth, churn_csv):
        headers_a, _ = auth
        ds = self._dataset(client, headers_a, churn_csv)
        r = client.post(
            "/api/auth/register",
            json={"email": "r2@test.pk", "display_name": "R", "password": "password-123"},
        )
        other = {"Authorization": f"Bearer {r.json()['access_token']}"}
        resp = client.post("/api/runs", json={"dataset_id": ds["id"]}, headers=other)
        assert resp.status_code == 404

    def test_bad_fold_count_is_422(self, client, auth, churn_csv):
        headers, _ = auth
        ds = self._dataset(client, headers, churn_csv)
        r = client.post(
            "/api/runs", json={"dataset_id": ds["id"], "n_folds": 99}, headers=headers
        )
        assert r.status_code == 422

    def test_predict_returns_one_row_per_input(self, client, auth, churn_csv):
        headers, _ = auth
        ds = self._dataset(client, headers, churn_csv)
        run = client.post(
            "/api/runs",
            json={"dataset_id": ds["id"], "n_folds": 3, "max_candidates": 2},
            headers=headers,
        ).json()

        rows = [
            {
                "tenure_months": 6,
                "monthly_bill_pkr": 3200,
                "support_calls": 5,
                "contract": "monthly",
                "has_internet": "yes",
            },
            {
                "tenure_months": 60,
                "monthly_bill_pkr": 1200,
                "support_calls": 0,
                "contract": "two_year",
                "has_internet": "no",
            },
        ]
        r = client.post(f"/api/runs/{run['id']}/predict", json={"rows": rows}, headers=headers)
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 2

    def test_predict_with_no_rows_is_422(self, client, auth, churn_csv):
        headers, _ = auth
        ds = self._dataset(client, headers, churn_csv)
        run = client.post(
            "/api/runs",
            json={"dataset_id": ds["id"], "n_folds": 3, "max_candidates": 2},
            headers=headers,
        ).json()
        r = client.post(f"/api/runs/{run['id']}/predict", json={"rows": []}, headers=headers)
        assert r.status_code == 422
