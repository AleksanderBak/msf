from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    epsilon: float = 1e-9
    instances_dir: str = Field(alias="INSTANCES_DIRECTORY")
    schedules_dir: str = Field(alias="SCHEDULES_DIRECTORY")
    visualization_dir: str = Field(alias="VISUALIZATION_DIRECTORY")


settings = Settings()
