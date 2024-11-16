import axios from "axios";

const PredictionService = {
    async predict(days: number): Promise<{
        prediction: number,
    }> {
        try{
            const prediction = await axios.get(`http://localhost:8000/api/v1/predict/${days}`);
            return prediction.data;
        } catch (error) {
            console.log(
                `Error found in ${__filename} - predict method: ${error.message}`,
            );
            throw(error);
        }
    }
}

export default PredictionService