import app_v3


def test_parse_metadata_xml_extracts_canonical_metadata_keys_from_real_xml():
    sample = b'''<?xml version="1.0" encoding="UTF-8"?>
<ns:Product xmlns:ns="http://pds.nasa.gov/pds4/pds/v1" xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2024-11-15T13:26:32.1339Z</start_date_time>
      <stop_date_time>2024-11-15T15:25:00.4388Z</stop_date_time>
    </Time_Coordinates>
    <Primary_Result_Summary>
      <processing_level>Calibrated</processing_level>
    </Primary_Result_Summary>
    <Investigation_Area>
      <name>Chandrayaan-2</name>
    </Investigation_Area>
    <Target_Identification>
      <name>South Pole</name>
    </Target_Identification>
    <Observing_System>
      <Observing_System_Component type="Spacecraft">
        <name>Chandrayaan-2</name>
      </Observing_System_Component>
      <Observing_System_Component type="Instrument">
        <name>orbiter high resolution camera</name>
      </Observing_System_Component>
    </Observing_System>
  </Observation_Area>
  <Mission_Area>
    <Product_Parameters>
      <imaging_orbit_number>23328</imaging_orbit_number>
    </Product_Parameters>
    <Geometry_Parameters>
      <spacecraft_altitude>93.67 km</spacecraft_altitude>
      <pixel_resolution>0.24 m/pixel</pixel_resolution>
      <roll>14.567189</roll>
      <pitch>11.610003</pitch>
      <yaw>0.012914</yaw>
      <sun_azimuth>180.299967</sun_azimuth>
      <sun_elevation>-0.169616</sun_elevation>
      <solar_incidence>90.169616</solar_incidence>
      <projection>Polar stereographic</projection>
      <area>South Pole</area>
      <focal_length>126.0</focal_length>
      <detector_pixel_width>19.0</detector_pixel_width>
      <Refined_Corner_Coordinates>
        <upper_left_latitude>-89.199860</upper_left_latitude>
        <upper_left_longitude>222.259328</upper_left_longitude>
        <upper_right_latitude>-89.209060</upper_right_latitude>
        <upper_right_longitude>229.728973</upper_right_longitude>
        <lower_left_latitude>-89.908842</lower_left_latitude>
        <lower_left_longitude>110.268353</lower_left_longitude>
        <lower_right_latitude>-89.946885</lower_right_latitude>
        <lower_right_longitude>22.453651</lower_right_longitude>
      </Refined_Corner_Coordinates>
    </Geometry_Parameters>
  </Mission_Area>
  <File_Area_Observational>
    <File>
      <file_name>sample.xml</file_name>
      <file_size>100</file_size>
    </File>
  </File_Area_Observational>
  <Array_2D_Image>
    <Element_Array>
      <data_type>UnsignedByte</data_type>
    </Element_Array>
    <Axis_Array>
      <Axis axis_name="Line">
        <elements>101074</elements>
      </Axis>
      <Axis axis_name="Sample">
        <elements>12000</elements>
      </Axis>
    </Axis_Array>
  </Array_2D_Image>
</ns:Product>'''

    meta = app_v3.parse_metadata_xml(sample)

    assert meta["sensor_type"] == "OHRC"
    assert meta["processing_level"] == "Calibrated"
    assert meta["gsd_m_per_pixel"] == 0.24
    assert meta["altitude_km"] == 93.67
    assert meta["footprint"]["upper_left"] == [-89.199860, 222.259328]
    assert meta["dimensions"]["lines"] == 101074
    assert meta["dimensions"]["samples"] == 12000
    assert meta["valid"] is True


def test_parse_metadata_xml_accepts_camel_case_and_ground_distance_gsd_tags():
    xml = b'''<Product>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2024-11-15T13:26:32Z</start_date_time>
      <stop_date_time>2024-11-15T13:26:48Z</stop_date_time>
    </Time_Coordinates>
    <Primary_Result_Summary><processing_level>Derived</processing_level></Primary_Result_Summary>
    <Observing_System>
      <Observing_System_Component type="Instrument"><name>terrain mapping camera</name></Observing_System_Component>
    </Observing_System>
  </Observation_Area>
  <Mission_Area>
    <Geometry_Parameters>
      <pixelResolution>5.0 m/pixel</pixelResolution>
      <spacecraft_altitude>110.59 km</spacecraft_altitude>
      <roll>0</roll><pitch>0</pitch><yaw>0</yaw>
      <sun_azimuth>1</sun_azimuth><sun_elevation>2</sun_elevation><solar_incidence>3</solar_incidence>
    </Geometry_Parameters>
  </Mission_Area>
</Product>'''

    meta = app_v3.parse_metadata_xml(xml)

    assert meta["sensor_type"] == "TMC"
    assert meta["gsd_m_per_pixel"] == 5.0
    assert meta["valid"] is False
    assert "Missing footprint coordinates" in meta["validation_errors"]


def test_parse_metadata_xml_extracts_gsd_from_comment_or_optical_formula():
    xml = b'''<Product>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-06-12T22:18:42Z</start_date_time>
      <stop_date_time>2023-06-12T22:28:35Z</stop_date_time>
    </Time_Coordinates>
    <Primary_Result_Summary><processing_level>Derived</processing_level></Primary_Result_Summary>
    <Observing_System>
      <Observing_System_Component type="Instrument"><name>terrain mapping camera</name></Observing_System_Component>
    </Observing_System>
  </Observation_Area>
  <Mission_Area>
    <Geometry_Parameters>
      <spacecraft_altitude>110.59 km</spacecraft_altitude>
      <focal_length unit="mm">140</focal_length>
      <detector_pixel_width unit="micrometer">7</detector_pixel_width>
      <roll>0</roll><pitch>0</pitch><yaw>0</yaw>
      <sun_azimuth>1</sun_azimuth><sun_elevation>2</sun_elevation><solar_incidence>3</solar_incidence>
    </Geometry_Parameters>
  </Mission_Area>
  <File_Area_Observational>
    <File>
      <comment>This File contains the ortho image derived data product with 5 meter resolution</comment>
    </File>
  </File_Area_Observational>
</Product>'''

    meta = app_v3.parse_metadata_xml(xml)
    assert meta["sensor_type"] == "TMC"
    assert meta["gsd_m_per_pixel"] == 5.0
    assert "description" in meta["gsd_source"]

