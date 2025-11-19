import React from "react";
import ReactMarkdown from "react-markdown";

const ResultCard = ({ result }) => {
  if (!result) return null;

  const { scientific_name, common_name, uses } = result;

  return (
    <div className="w-full max-w-2xl bg-white/90 backdrop-blur-xl shadow-xl border border-gray-200 rounded-2xl p-8 mt-8">
      <h2 className="text-3xl font-bold text-green-700 mb-4">
        🌿 Plant Details
      </h2>

      <div className="space-y-2 text-lg text-gray-700">
        <p>
          <strong>Scientific Name:</strong>{" "}
          <span className="text-green-700">{scientific_name}</span>
        </p>
        <p>
          <strong>Common Name:</strong>{" "}
          <span className="text-green-700">{common_name}</span>
        </p>
      </div>

      <h3 className="text-2xl font-semibold text-gray-800 mt-6 mb-3">
        Medicinal Uses 🌱
      </h3>

      <div className="prose prose-green bg-green-50 p-5 rounded-xl shadow-inner leading-relaxed">
        <ReactMarkdown>{uses}</ReactMarkdown>
      </div>
    </div>
  );
};

export default ResultCard;
