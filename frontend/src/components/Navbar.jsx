import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="bg-white shadow-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
      <h1 className="text-2xl font-bold text-green-700">🌿 AyurDrishti</h1>

      <div className="flex gap-6 text-lg">
        <Link
          to="/"
          className="hover:text-green-600 font-medium transition-all duration-200"
        >
          Identify Plant
        </Link>

        <Link
          to="/chat"
          className="hover:text-green-600 font-medium transition-all duration-200"
        >
          Ayurveda Chatbot
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
