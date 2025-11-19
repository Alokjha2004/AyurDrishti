import { useState } from "react";
import ImageUpload from "../components/ImageUpload";
import ResultCard from "../components/ResultCard";

const Home = () => {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-100 to-green-300 flex flex-col items-center py-10 px-4">
      
      <h1 className="text-4xl font-extrabold text-green-800 mb-8 drop-shadow-md">
        Ayur Drishti 🌿
      </h1>

      <ImageUpload setResult={setResult} />
      <ResultCard result={result} />
      
    </div>
  );
};

export default Home;
