from pydantic import BaseModel, Field


class PageRequest(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class PagedResponse[T](BaseModel):
    page: int
    page_size: int
    total: int
    records: list[T]
