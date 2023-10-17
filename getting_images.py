import argparse
import nasapy
import os
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd

def setup_argparse():
    parser = argparse.ArgumentParser(description="NASA Image and Video Library Dataset")
    parser.add_argument("-k", "--api-key", required=True, help="NASA API key")
    parser.add_argument("--exoplanet", action="store_true", help="Get and save exoplanet images")
    parser.add_argument("--planet-concept", action="store_true", help="Get and save planet artist concept images")
    parser.add_argument("--planet-photographs", action="store_true", help="Get and save planet photographs images")
    return parser

def get_images(database_name):
    for i, image in enumerate(database_name):
        link_data = image['links']

        for url in link_data:
            image_url = url['href']
            image_to_show = io.imread(image_url)
            plt.imshow(image_to_show)
            plt.xlabel(i)
            plt.show()

def save_keep_images(database_name, keep_indexes):
    keep_images = []
    for i, image in enumerate(database_name):
        if i in keep_indexes:
            keep_images.append(database_name[i])
    return keep_images

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    api_key = args.api_key
    nasa = nasapy.Nasa(key=api_key)

    if args.exoplanet:
        exoplanet_images = nasa.media_search(query="exoplanet", media_type="image")
        exoplanet_data = exoplanet_images['items']
        get_images(exoplanet_data)

        keep_indexes = [0, 35, 36, 37, 38, 39, 41, 42, 45, 46, 54, 55, 65]
        save_keep_images(exoplanet_data, keep_indexes)

    if args.planet_concept:
        planet_images = nasa.media_search(query="planet artist concept", media_type="image")
        planet_data = planet_images["items"]
        get_images(planet_data)

        keep_indexes = [0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 25, 26, 35, 43, 45, 46, 47, 48, 51, 52, 53, 54, 55, 57, 59, 62, 69, 70, 71, 72, 73, 74, 76, 77, 80, 83, 84, 85, 86, 87, 92, 96]
        save_keep_images(planet_data, keep_indexes)

    if args.planet_photographs:
        planet_photos = nasa.media_search(query="planet photographs", media_type="image")
        planet_photo_data = planet_photos["items"]
        get_images(planet_photo_data)

        keep_indexes = [4, 6, 14, 29, 77, 80, 87]
        save_keep_images(planet_photo_data, keep_indexes)

if __name__ == "__main__":
    main()

