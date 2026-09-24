"""Stdio-to-Runner bridge; has no Docker access, tools or application secrets."""
import asyncio
import json
import sys

import websockets


async def main():
    mode, socket_path, owner_id, run_id = sys.argv[1:5]
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    async with websockets.unix_connect(socket_path, uri="ws://localhost/mcp", max_size=16 * 1024 * 1024) as channel:
        await channel.send(json.dumps({"mode": mode, "owner_id": owner_id or None,
                                       "run_id": run_id or None}))

        async def send_input():
            while line := await reader.readline():
                await channel.send(line)

        async def receive_output():
            async for chunk in channel:
                if not isinstance(chunk, bytes):
                    raise RuntimeError("Invalid MCP frame")
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

        tasks = [asyncio.create_task(send_input()), asyncio.create_task(receive_output())]
        done, waiting = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)
        for task in done:
            task.result()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        # Do not print the socket path, document contents or MCP parameters.
        sys.stderr.write("MCP Runner bridge unavailable\n")
        raise SystemExit(1) from None
