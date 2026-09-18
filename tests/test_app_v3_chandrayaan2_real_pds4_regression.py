import json
from pathlib import Path

import app_v3


def test_parse_chandrayaan2_pds4_xml_real_files_match_expected_contract():
    source_path = Path("tests/fixtures/chandrayaan2/ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml")
    reference_path = Path("tests/fixtures/chandrayaan2/ch2_ohr_ncp_20241115T1525004388_d_img_d18.xml")

    assert source_path.exists(), "Missing source test fixture: ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml"
    assert reference_path.exists(), "Missing reference test fixture: ch2_ohr_ncp_20241115T1525004388_d_img_d18.xml"

    source_bytes = source_path.read_bytes()
    reference_bytes = reference_path.read_bytes()

    source = app_v3.parse_chandrayaan2_pds4_xml(source_bytes)
    reference = app_v3.parse_chandrayaan2_pds4_xml(reference_bytes)

    # 1. Sensor & Instrument
    assert source["sensor_type"] == "OHRC"
    assert reference["sensor_type"] == "OHRC"
    assert source["instrument"] == "orbiter high resolution camera"
    assert reference["instrument"] == "orbiter high resolution camera"

    # 2. Processing Level & Times
    assert source["processing_level"] == "Calibrated"
    assert reference["processing_level"] == "Calibrated"
    assert source["start_time"] == "2024-11-15T13:26:32.1339Z"
    assert reference["start_time"] == "2024-11-15T15:25:00.4388Z"

    # 3. GSD & Altitude (numeric floats)
    assert source["gsd_m_per_pixel"] == 0.24
    assert isinstance(source["gsd_m_per_pixel"], float)
    assert reference["gsd_m_per_pixel"] == 0.24
    assert isinstance(reference["gsd_m_per_pixel"], float)

    assert source["altitude_km"] == 93.67
    assert isinstance(source["altitude_km"], float)
    assert reference["altitude_km"] == 93.46
    assert isinstance(reference["altitude_km"], float)

    # 4. Geometry & Sun Angles
    assert source["roll_deg"] == 14.567189
    assert source["pitch_deg"] == 11.610003
    assert source["yaw_deg"] == 0.012914
    assert source["sun_azimuth_deg"] == 180.299967
    assert source["sun_elevation_deg"] == -0.169616
    assert source["solar_incidence_deg"] == 90.169616

    # 5. Projection & Area
    assert source["projection"] == "Polar stereographic"
    assert reference["projection"] == "Polar stereographic"
    assert source["area"] == "South Pole"
    assert reference["area"] == "South Pole"

    # 6. Refined Footprint Corners
    assert source["footprint"]["upper_left"] == [-89.199860, 222.259328]
    assert source["footprint"]["upper_right"] == [-89.209060, 229.728973]
    assert source["footprint"]["lower_left"] == [-89.908842, 110.268353]
    assert source["footprint"]["lower_right"] == [-89.946885, 22.453651]

    assert reference["footprint"]["upper_left"] == [-89.196875, 221.909032]

    # 7. Dimensions & Data Type
    assert source["dimensions"]["lines"] == 101074
    assert source["dimensions"]["samples"] == 12000
    assert source["data_type"] == "UnsignedByte"

    assert reference["dimensions"]["lines"] == 101074
    assert reference["dimensions"]["samples"] == 12000
    assert reference["data_type"] == "UnsignedByte"

    # 8. Validation Status
    assert source["valid"] is True
    assert source["validation_errors"] == []
    assert reference["valid"] is True
    assert reference["validation_errors"] == []


def test_parse_chandrayaan2_pds4_xml_child_type_and_axis_array_elements():
    xml_content = b"""<?xml version="1.0" encoding="UTF-8"?>
<Product xmlns="http://pds.nasa.gov/pds4/pds/v1" xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
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
      <Observing_System_Component>
        <name>Chandrayaan 2 Orbiter</name>
        <type>Spacecraft</type>
      </Observing_System_Component>
      <Observing_System_Component>
        <name>orbiter high resolution camera</name>
        <type>Instrument</type>
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
  <Array_2D_Image>
    <Element_Array>
      <data_type>UnsignedByte</data_type>
    </Element_Array>
    <Axis_Array>
      <axis_name>Line</axis_name>
      <elements>101074</elements>
    </Axis_Array>
    <Axis_Array>
      <axis_name>Sample</axis_name>
      <elements>12000</elements>
    </Axis_Array>
  </Array_2D_Image>
</Product>"""
    meta = app_v3.parse_chandrayaan2_pds4_xml(xml_content)
    assert meta["sensor_type"] == "OHRC"
    assert meta["instrument"] == "orbiter high resolution camera"
    assert meta["dimensions"]["lines"] == 101074
    assert meta["dimensions"]["samples"] == 12000
    assert meta["data_type"] == "UnsignedByte"
    assert meta["valid"] is True
    assert meta["validation_errors"] == []


