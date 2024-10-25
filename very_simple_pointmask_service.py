import os,io,base64,tempfile
from fastapi import FastAPI, APIRouter
from pm.service_api.pm_service import pm_router
API_VERSION_PREFIX="/v1"

hello_router = APIRouter(prefix="/hello")
@hello_router.get("/test")
def get():
    return {"hello":"hello"}


app = FastAPI()
app.include_router(hello_router, prefix=API_VERSION_PREFIX)
app.include_router(pm_router,prefix=API_VERSION_PREFIX)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,port=8001)