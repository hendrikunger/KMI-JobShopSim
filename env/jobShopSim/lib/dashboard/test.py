import websockets
from websockets.client import connect
import asyncio
from plotly.graph_objs._figure import Figure as PlotlyFigure

async def hello():
    uri = "ws://127.0.0.1:5000"
    fig = PlotlyFigure()
    async with connect(uri) as websocket:
        data = plotly.io.to_json(fig=fig)
        await websocket.send(data)
        #print(f">>> {name}")
        #greeting = await websocket.recv()
        #print(f"<<< {greeting}")
        
if __name__ == "__main__":
    asyncio.run(hello())