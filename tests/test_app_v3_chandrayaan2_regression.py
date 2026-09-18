import app_v3


def test_parse_metadata_xml_handles_actual_chandrayaan2_ohrc_structure():
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
<Metadata>
  <Product>
    <Sensor>OHRC</Sensor>
    <ProductLevel>Level2</ProductLevel>
  </Product>
  <Image>
    <Latitude>12.345</Latitude>
    <Longitude>67.890</Longitude>
    <ResolutionGSD>0.24</ResolutionGSD>
    <Altitude>123.4</Altitude>
    <SunAzimuth>45.5</SunAzimuth>
    <SunElevation>12.3</SunElevation>
    <SolarIncidence>34.0</SolarIncidence>
    <Roll>1.1</Roll>
    <Pitch>2.2</Pitch>
    <Yaw>3.3</Yaw>
    <ImageWidth>512</ImageWidth>
    <ImageHeight>512</ImageHeight>
    <Projection>WGS84</Projection>
  </Image>
  <Footprint>
    <Corner1>12.345,67.890</Corner1>
    <Corner2>12.350,67.895</Corner2>
    <Corner3>12.355,67.900</Corner3>
    <Corner4>12.360,67.905</Corner4>
  </Footprint>
</Metadata>'''

    meta = app_v3.parse_metadata_xml(xml)

    assert meta["sensor_type"] == "OHRC"
    assert meta["processing_level"] == "Level2"
    assert meta["image_width"] == "512"
    assert meta["image_height"] == "512"
    assert meta["projection"] == "WGS84"
    assert len(meta.get("footprint_corners", [])) == 4
