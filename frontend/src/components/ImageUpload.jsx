import React, { useState } from "react";
import { API } from "../api/api";

const ImageUpload = ({ setResult }) => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImage = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image");

    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await API.post("/predict", formData);
      setResult(res.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-xl p-6 bg-white/80 backdrop-blur-xl shadow-xl rounded-2xl border border-gray-200">
      <h2 className="text-2xl font-bold text-center text-green-700 mb-5">
        Upload Leaf Image 🍃
      </h2>

      <label className="w-full flex flex-col items-center p-6 border-2 border-dashed border-green-400 rounded-xl cursor-pointer bg-green-50 hover:bg-green-100 transition">
        <span className="text-gray-600 mb-3">Click to choose an image</span>
        <input type="file" onChange={handleImage} className="hidden" />
      </label>

      {preview && (
        <img
          src={preview}
          alt="preview"
          className="w-56 h-56 object-cover rounded-xl mx-auto mt-5 shadow-md"
        />
      )}

      <button
        onClick={handleUpload}
        disabled={loading}
        className={`w-full mt-6 py-3 text-white font-bold rounded-xl
          ${loading ? "bg-gray-400" : "bg-green-600 hover:bg-green-700"}
          transition shadow-lg`}
      >
        {loading ? "Identifying..." : "Identify Plant 🌿"}
      </button>
    </div>
  );
};

export default ImageUpload;
