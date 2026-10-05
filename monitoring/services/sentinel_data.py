from collections import defaultdict

from pystac_client import Client


STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"


def search_sentinel_images(
    bbox,
    start_date,
    end_date,
    max_cloud=20,
    max_items=100,
):
    """
    Search Sentinel-2 L2A imagery
    for bbox and date interval.
    """

    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud
            }
        },
        max_items=max_items,
    )

    items = list(search.items())

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100
        )
    )

    return items


def get_tile_id(item):
    """
    Example:
    S2B_T40TFN_20260624T070638_L2A
        -> T40TFN
    """

    parts = item.id.split("_")

    if len(parts) >= 2:
        return parts[1]

    return "UNKNOWN"


def group_by_tile(items):
    """
    Group STAC items by Sentinel-2 tile.
    """

    groups = defaultdict(list)

    for item in items:
        tile = get_tile_id(item)
        groups[tile].append(item)

    return groups


def print_results(items):
    """
    Print all found scenes.
    """

    if not items:
        print("No Sentinel-2 images found.")
        return

    print(
        f"\nFound {len(items)} Sentinel-2 images:\n"
    )

    for index, item in enumerate(
        items,
        start=1
    ):

        cloud = item.properties.get(
            "eo:cloud_cover",
            "N/A"
        )

        tile = get_tile_id(item)

        print(
            f"{index}. "
            f"{item.datetime} | "
            f"Cloud: {cloud}% | "
            f"Tile: {tile} | "
            f"ID: {item.id}"
        )


def print_tiles(items):
    """
    Print tile summary.
    """

    groups = group_by_tile(items)

    print("\nAVAILABLE TILES:\n")

    for tile, tile_items in groups.items():

        best = min(
            tile_items,
            key=lambda item:
                item.properties.get(
                    "eo:cloud_cover",
                    100
                )
        )

        cloud = best.properties.get(
            "eo:cloud_cover",
            "N/A"
        )

        print(
            f"{tile}: "
            f"{len(tile_items)} scenes | "
            f"best cloud = {cloud}%"
        )