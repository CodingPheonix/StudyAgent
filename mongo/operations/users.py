import bcrypt

from mongo.init import users
from mongo.schema import User

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(hashed: str, password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def register(name: str, email: str, password: str):
    # Hash incomming password
    hashed = hash_password(password)

    # Check if user exists
    existing_user = users.find_one({"email": email})

    if not existing_user:
        # Create new User Object
        new_user = User(name=name, email=email, password=hashed)

        # Insert new User
        data = users.insert_one(new_user.model_dump())

        return {"status": 201, "data": {
            "name": name,
            "email": email,
            "id": str(data.inserted_id)
        }}
    else:
        return {"status": 200}

def user_login(email: str, password: str):
    hashed = hash_password(password)
    existing_user = users.find_one({"email": email})

    if not check_password(hashed, password) or existing_user is None:
        return {"status": 404}
    else:
        return {"status": 200, "data": {
            "name": existing_user["name"],
            "email": existing_user["email"],
            "id": str(existing_user["_id"])
        }}
