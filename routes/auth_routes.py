import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import auth
import models
from schemas import UserCreate, Token

logger = logging.getLogger("neurobuilder.auth")
router = APIRouter(tags=["auth"])


@router.post("/register")
def register(user: UserCreate, db: Session = Depends(auth.get_db)):
    try:
        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if db_user:
            logger.warning("Registration failed: Email %s already registered", user.email)
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed_password = auth.get_password_hash(user.password)
        new_user = models.User(full_name=user.full_name, email=user.email, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info("Successfully registered new user: %s", user.email)
        return {"message": "User created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during registration for %s: %s", user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(auth.get_db)):
    try:
        user = db.query(models.User).filter(models.User.email == form_data.username).first()
        if not user or not auth.verify_password(form_data.password, user.hashed_password):
            logger.warning("Login failed for user: %s", form_data.username)
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = auth.create_access_token(data={"sub": user.email})
        logger.info("User %s successfully logged in", user.email)
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during login for %s: %s", form_data.username, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
