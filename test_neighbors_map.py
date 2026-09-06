import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt

gdf_india = gpd.read_file('india_states.geojson')
india_union = unary_union(gdf_india.geometry)

shape_path = 'venv/Lib/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'
gdf_world = gpd.read_file(shape_path)

# Neighbors
neighbor_names = ['Pakistan', 'China', 'Nepal', 'Bhutan', 'Bangladesh', 'Myanmar', 'Sri Lanka', 'Afghanistan']
gdf_neighbors = gdf_world[gdf_world['name'].isin(neighbor_names)].copy()
gdf_neighbors['geometry'] = gdf_neighbors['geometry'].apply(lambda g: g.difference(india_union))

print("Bounds of India:", gdf_india.total_bounds)
print("Neighbors extracted successfully!")
