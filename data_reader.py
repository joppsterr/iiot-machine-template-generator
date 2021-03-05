import asyncio
import logging
from datetime import datetime

import csv
import aiofiles
from aiocsv import AsyncDictWriter

from asyncua import Client

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('asyncua')

async def main():
    # TCP connection own server
    url = 'opc.tcp://0.0.0.0:4840/freeopcua/server/'
    file_name = 'dataset.csv'
    # Prepare csv file
    async with aiofiles.open(file_name, mode="w", encoding="utf-8", newline="") as afp:
        writer = AsyncDictWriter(afp, ["time", "value"], restval="NULL", quoting=csv.QUOTE_ALL)
        await writer.writeheader()

    async with Client(url=url) as client:
        while True:
            # Do stuff
            # Get  nodes
            # TODO: Get info from all nodes
            node = client.get_node("ns=2;i=2")
            value = await node.read_value()
            async with aiofiles.open(file_name, mode="a", encoding="utf-8", newline="") as afp:
                writer = AsyncDictWriter(afp, ["time", "value"], restval="NULL", quoting=csv.QUOTE_ALL)
                await writer.writerow({"time": datetime.timestamp(datetime.now()), "value": value})
            
            await asyncio.sleep(1)
            
                

if __name__ == '__main__':
    asyncio.run(main())
