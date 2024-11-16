import {GasPriceModel} from "./model";

const GraphService = {
    async getData(): Promise<boolean> {
        try {
            const oneWeekAgo = new Date();
            oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
            const groupedData = await GasPriceModel.aggregate([
                {
                    // Filter records to only include those from the last 7 days
                    $match: {
                        timestamp: {$gte: oneWeekAgo}
                    }
                },
                {
                    // Group by the 'day' field
                    $group: {
                        _id: "$day", // Group by the 'day' field (which is a string, e.g., YYYY-MM-DD)
                        totalGasPrice: {$avg: "$gas_price_gwei"}, // Example: calculate average gas price for the day
                        count: {$sum: 1}, // Example: count the number of entries for that day
                        records: {$push: "$$ROOT"} // Collect all records for that day
                    }
                },
                {
                    // Sort the grouped data by 'timestamp' field (time), in ascending order
                    $unwind: "$records" // Unwind to access the individual documents inside the 'records' array
                },
                {
                    $sort: {"records.timestamp": 1} // Sort records within each group by timestamp (ascending)
                },
                {
                    // Optional: Re-group by 'day' after sorting
                    $group: {
                        _id: "$_id", // Group by day again
                        totalGasPrice: {$first: "$totalGasPrice"}, // Take the average from the first group
                        count: {$first: "$count"}, // Take the count from the first group
                        records: {$push: "$records"} // Reassemble the sorted records into an array
                    }
                },
                {
                    // Optional: Sort by the 'day' field in descending order
                    $sort: {"_id": -1}
                }
            ]);

            // console.log(groupedData);

            const gasPricesLastWeek = await GasPriceModel.find();
            console.log(gasPricesLastWeek);

            return true;

        } catch (error) {
            console.log(
                `Error found in ${__filename} - getData method: ${error.message}`,
            );
            throw (error);
        }
    }
}

export default GraphService