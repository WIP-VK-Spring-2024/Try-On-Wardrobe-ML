"""Module for decorator `collect_response`."""
import json

import aio_pika
import pydantic

from app.internal.repository.rabbitmq.handlers.handle_exception import handle_exception

__all__ = ["load_response"]


# TODO: add support collect response for async generator
@handle_exception
async def load_response(response, model):
    response = json.loads(response.body.decode())

    return pydantic.parse_obj_as(
        model,
        await __convert_response(response=response, annotations=str(model)),
    )


async def __convert_response(response: dict, annotations: str):
    """Converts the response of the request to List of models or to a single
    model.

    Args:
        response: Response of aioredis query.
        annotations: Annotations of `fn`.

    Returns: List[`Model`] if List is specified in the type annotations,
            or a single `Model` if `Model` is specified in the type annotations.
    """
    r = response.copy()

    if annotations.replace("typing.", "").startswith("List"):
        return [await __convert_memory_viewer(i) for i in r.values()]
    return await __convert_memory_viewer(r)


async def __convert_memory_viewer(r: dict):
    """Convert memory viewer in bytes.

    Notes: aioredis returns memory viewer in query response,
        when in database type of cell `bytes`.
    """

    for key, value in r.items():
        if isinstance(value, memoryview):
            r[key] = value.tobytes()
    return r
