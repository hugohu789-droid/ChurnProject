import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_register_success(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "test@example.com", "username": "testuser", "password": "securepass"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["username"] == "testuser"


async def test_register_duplicate_email(client: AsyncClient):
    payload = {"email": "dup@example.com", "username": "user1", "password": "securepass"}
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post(
        "/api/v1/auth/register",
        json={**payload, "username": "user2"},
    )
    assert response.status_code == 400


async def test_login_success(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "login@example.com", "username": "loginuser", "password": "securepass"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "securepass"},
    )
    assert response.status_code == 200
    tokens = response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"


async def test_login_wrong_password(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpw@example.com", "username": "wpuser", "password": "securepass"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpw@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


async def test_me_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 403


async def test_me_with_valid_token(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "me@example.com", "username": "meuser", "password": "securepass"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "me@example.com", "password": "securepass"},
    )
    token = login.json()["access_token"]

    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
