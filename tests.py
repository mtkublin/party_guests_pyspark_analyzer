import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as st

from party_guests import calculate_peak_guests_amount, get_peak_amount_guest_ids


@pytest.fixture(scope="session")
def spark_test_session():
    spark = SparkSession.builder.master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    return spark


@pytest.fixture(scope="session")
def input_schema():
    schema = st.StructType(
        [
            st.StructField("guest_id", st.IntegerType(), False),
            st.StructField("entrance_date", st.StringType(), False),
            st.StructField("exit_date", st.StringType(), False),
        ]
    )
    return schema


TESTS_INPUT_DATA = [
    # all guests coming right after previous one left -> each guest creates a separate peak
    [
        (0, "2020-12-31 23:59:00", "2020-12-31 23:59:59"),
        (1, "2021-01-01 00:00:00", "2021-01-01 00:00:59"),
        (2, "2021-01-01 00:01:00", "2021-01-01 00:01:59"),
        (3, "2021-01-01 00:02:00", "2021-01-01 00:02:59"),
        (4, "2021-01-01 00:03:00", "2021-01-01 00:03:59"),
    ],
    # 3 separate peaks with 3 guests present at each (as on each leave from the 1st peak a new guest arrives)
    [
        (0, "2021-01-01 00:00:00", "2021-01-01 00:15:00"),
        (1, "2021-01-01 00:05:00", "2021-01-01 00:20:00"),
        (2, "2021-01-01 00:10:00", "2021-01-01 00:25:00"),
        (3, "2021-01-01 00:15:00", "2021-01-01 00:30:00"),
        (4, "2021-01-01 00:20:00", "2021-01-01 00:35:00"),
    ],
    # single peak after first 3 guests arrive
    [
        (0, "2021-01-01 00:00:00", "2021-01-01 00:15:00"),
        (1, "2021-01-01 00:05:00", "2021-01-01 00:20:00"),
        (2, "2021-01-01 00:10:00", "2021-01-01 00:25:00"),
        (3, "2021-01-01 00:20:01", "2021-01-01 00:35:00"),
        (4, "2021-01-01 00:25:01", "2021-01-01 00:40:00"),
    ],
    # single peak after all guests arrive
    [
        (0, "2021-01-01 00:00:00", "2021-01-01 00:30:00"),
        (1, "2021-01-01 00:05:00", "2021-01-01 00:35:00"),
        (2, "2021-01-01 00:10:00", "2021-01-01 00:40:00"),
        (3, "2021-01-01 00:15:00", "2021-01-01 00:45:00"),
        (4, "2021-01-01 00:20:00", "2021-01-01 00:50:00"),
    ],
    # 3 separate peaks with 3 guests present at each
    [
        (0, "2021-01-01 00:00:00", "2021-01-01 01:30:13"),
        (1, "2021-01-01 00:15:00", "2021-01-01 00:45:00"),
        (2, "2021-01-01 00:16:00", "2021-01-01 04:32:00"),
        (3, "2021-01-01 01:00:00", "2021-01-01 03:30:00"),
        (4, "2021-01-01 01:34:00", "2021-01-01 01:56:00"),
    ],
]


@pytest.mark.parametrize(
    ("input_data", "expected_result"),
    zip(TESTS_INPUT_DATA, [1, 3, 3, 5, 3]),
)
def test_calculate_peak_guests_amount(spark_test_session, input_schema, input_data, expected_result):
    input_df = spark_test_session.createDataFrame(data=input_data, schema=input_schema)
    actual_result = calculate_peak_guests_amount(party_guests=input_df)

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_data", "expected_result"),
    zip(
        TESTS_INPUT_DATA,
        [
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},
            {0, 1, 2},
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},
        ],
    ),
)
def test_get_peak_amount_guest_ids(spark_test_session, input_schema, input_data, expected_result):
    input_df = spark_test_session.createDataFrame(data=input_data, schema=input_schema)
    actual_result = get_peak_amount_guest_ids(party_guests=input_df)

    assert actual_result == expected_result
