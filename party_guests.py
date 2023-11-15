from pyspark.sql import DataFrame, Window, functions as sf


def calculate_peak_guests_amount(party_guests: DataFrame) -> int:
    actions = _calculate_cumulative_guests_amount(party_guests)
    max_amount = actions.agg({"cumulative_guests_amount": "max"})
    result = max_amount.collect()[0][0]
    return result


def get_peak_amount_guest_ids(party_guests: DataFrame) -> set[int]:
    actions = _calculate_cumulative_guests_amount(party_guests)
    peaks = _get_peaks(actions)
    # multiple peaks possible, we need to get ids of guests present at each of them
    peak_actions = actions.join(peaks, on=[actions.datetime <= peaks.peak_time])
    guests_present_at_peaks = _get_guests_present_at_peaks(peak_actions)
    result = guests_present_at_peaks.rdd.map(lambda x: x.guest_id).collect()
    return set(result)


def _calculate_cumulative_guests_amount(party_guests: DataFrame) -> DataFrame:
    entrances = _transform_guests_list(party_guests, "entrance_date", 1)
    exits = _transform_guests_list(party_guests, "exit_date", -1)
    all_actions = entrances.union(exits)

    window_spec = (
        Window
        .partitionBy(sf.lit(0))
        .orderBy('datetime')
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    all_actions = all_actions.withColumn(
        "cumulative_guests_amount",
        sf.sum("guests_amount_change").over(window_spec)
    )
    return all_actions


def _transform_guests_list(party_guests: DataFrame, date_column: str, guests_amount: int) -> DataFrame:
    return party_guests.select(
        sf.col("guest_id"),
        sf.to_timestamp(date_column, "yyyy-MM-dd HH:mm:ss").alias("datetime"),
        sf.lit(guests_amount).alias("guests_amount_change"),
    )


def _get_peaks(actions: DataFrame) -> DataFrame:
    """
    We need to select all possible peaks, are there might be many.
    """
    peak_window_spec = Window.partitionBy(sf.lit(0))
    peaks = (
        actions
        .withColumn(
            "max_guests",
            sf.max("cumulative_guests_amount").over(peak_window_spec)
        )
        .where(sf.col("max_guests") == sf.col("cumulative_guests_amount"))
        .select(
            sf.col("datetime").alias("peak_time")
        )
        .distinct()
    )
    return peaks


def _get_guests_present_at_peaks(peak_actions: DataFrame) -> DataFrame:
    """
    We sum guests amount change per each guest + peak time pair, as this will result in either 0 (if the guest exited
    before the peak time) or 1 (if they stayed). Then we select only guests who stayed and select their distinct ids.
    """
    guest_presence_window_spec = Window.partitionBy("guest_id", "peak_time")
    guests_present_at_peaks = (
        peak_actions
        .withColumn(
            "guest_presence",
            sf.sum("guests_amount_change").over(guest_presence_window_spec)
        )
        .where(sf.col("guest_presence") == sf.lit(1))
        .select("guest_id")
        .distinct()
    )
    return guests_present_at_peaks
