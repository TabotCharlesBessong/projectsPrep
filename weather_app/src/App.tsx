import React, { useState } from "react";
import { motion } from "framer-motion";
import { WeatherInfo } from "./constant/types";
import { weatherData } from "./constant/data";
import { CiRainbow } from "react-icons/ci";
import { BsCloudDrizzle, BsSunrise } from "react-icons/bs";
import { BiCloud } from "react-icons/bi";
import { LuHaze } from "react-icons/lu";


const App: React.FC = () => {
  const [city, setCity] = useState<string>("");
  const [result, setResult] = useState<WeatherInfo | null>(null);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  const handleSearch = () => {
    const formattedCity = city.trim();
    console.log(weatherData[formattedCity])
    if (formattedCity && weatherData[formattedCity]) {
      setResult(weatherData[formattedCity]);
      setSearchHistory((prev) => {
        const updatedHistory = [
          formattedCity,
          ...prev.filter((c) => c !== formattedCity),
        ];
        return updatedHistory.slice(0, 10);
      });
    } else {
      alert("City not found in the database!");
    }
    // setCity("");
  };

  const handleHistoryClick = (city: string) => {
    setResult(weatherData[city]);
  };

  return (
    <div className="min-h-screen bg-blue-100 flex flex-col items-center p-6">
      <motion.h1
        className="text-3xl font-bold text-blue-600 mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Weather Application
      </motion.h1>

      <div className="w-full max-w-md">
        <div className="flex items-center gap-2 mb-4">
          <input
            type="text"
            className="flex-grow px-4 py-2 border border-gray-300 rounded shadow"
            placeholder="Enter city name"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
          <button
            onClick={handleSearch}
            className="px-4 py-2 bg-blue-500 text-white font-semibold rounded shadow hover:bg-blue-600"
          >
            Search
          </button>
        </div>

        {result && (
          <motion.div
            className="p-4 bg-white rounded shadow mb-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h2 className="text-xl font-bold mb-2">Weather in {city}</h2>
            <p>Temperature: {result.temperature}</p>
            <p>Humidity: {result.humidity}</p>
            <p>Wind Speed: {result.windSpeed}</p>
            <p>
              Weather Condition: {result.condition}{" "}
              {result.condition === "Rainy" ? (
                <CiRainbow size={32} color="black" className="inline ml-4" />
              ) : result.condition === "Sunny" ? (
                <BsSunrise size={32} color="black" className="inline ml-4" />
              ) : result.condition === "Cloudy" ? (
                <BiCloud size={32} color="black" className="inline ml-4" />
              ) : result.condition === "Partly Cloudy" ? (
                <BsCloudDrizzle size={32} color="black" className="inline ml-4" />
              ) : (
                <LuHaze size={32} color="black" className="inline ml-4" />
              )}{" "}
            </p>
            <p>Visibility: {result.visibility} </p>
            <p>Pressure: {result.pressure}</p>
          </motion.div>
        )}

        {searchHistory.length > 0 && (
          <div>
            <h3 className="text-lg font-bold mb-2">Search History</h3>
            <ul className="space-y-2">
              {searchHistory.map((cityName, index) => (
                <motion.li
                  key={index}
                  className="cursor-pointer text-blue-500 hover:underline"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.1 }}
                  onClick={() => handleHistoryClick(cityName)}
                >
                  {cityName}
                </motion.li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
