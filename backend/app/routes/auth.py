from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import hashlib
import secrets
from typing import Optional
from fastapi import Header

from ..database import get_db
from ..models import User
from ..schemas import UserRegister, UserLogin, AuthResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["User Authentication"])

def hash_password(password: str) -> tuple[str, str]:
    """
    Hash a password using PBKDF2-SHA256 with a unique salt.
    Returns (hashed_password_hex, salt_hex).
    """
    salt = secrets.token_hex(16)
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    return dk.hex(), salt

def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against a stored hash and salt.
    """
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    return dk.hex() == hashed_password

def generate_session_token(user_id: int) -> str:
    """
    Generate a session token containing the user ID and a random hex string.
    """
    return f"user-session-{user_id}-{secrets.token_hex(16)}"

@router.post("/register", response_model=AuthResponse)
def register_user(request: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user, hashes their password, and logs them in immediately.
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email.lower()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists."
        )

    hashed_pw, salt = hash_password(request.password)
    
    new_user = User(
        email=request.email.lower(),
        hashed_password=hashed_pw,
        password_salt=salt,
        full_name=request.full_name
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        token = generate_session_token(new_user.id)
        user_response = UserResponse.model_validate(new_user)
        
        return {
            "user": user_response,
            "token": token
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )

@router.post("/login", response_model=AuthResponse)
def login_user(request: UserLogin, db: Session = Depends(get_db)):
    """
    Validate user credentials and return a session token.
    """
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
        
    if not verify_password(request.password, user.hashed_password, user.password_salt):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
        
    token = generate_session_token(user.id)
    user_response = UserResponse.model_validate(user)
    
    return {
        "user": user_response,
        "token": token
    }

def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    """
    Dependency to get the currently authenticated user from the Authorization header.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Authorization header missing."
        )
        
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication header format. Use 'Bearer <token>'."
        )
        
    token = authorization.split(" ")[1]
    if not token.startswith("user-session-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token."
        )
        
    try:
        parts = token.split("-")
        # token format: user-session-{id}-{random_hex}
        user_id = int(parts[2])
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account associated with this session was not found."
            )
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token structure."
        )
