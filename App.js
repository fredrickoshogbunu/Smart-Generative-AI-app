import React, { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [responseText, setResponseText] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle form submission by calling the API with Axios
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResponseText("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        // Handle HTTP errors
        const errorData = await response.json();
        throw new Error(errorData.detail?.error || "Server error occurred");
      }

      const data = await response.json();

      if (data && data.response) {
        setResponseText(data.response);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error("API Error:", error);
      setError(error.message || "Failed to get response from AI service");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1>Smart GenAI Query Interface</h1>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query"
          style={styles.input}
        />
        <button type="submit" style={styles.button} disabled={loading}>
          {loading ? "Submitting..." : "Submit"}
        </button>
      </form>

      <div style={styles.responseContainer}>
        <h2>Response:</h2>
        {error && <p style={styles.errorText}>{error}</p>}
        <pre style={styles.responseText}>{responseText}</pre>
      </div>
    </div>
  );
}

// CSS-in-JS styles
const styles = {
  container: {
    padding: "20px",
    backgroundColor: "#FFDAB9", // Peach background
    minHeight: "100vh",
    textAlign: "center",
  },
  form: {
    marginTop: "20px",
  },
  input: {
    width: "300px",
    marginRight: "10px",
    padding: "8px",
    fontSize: "16px",
  },
  button: {
    padding: "8px 15px",
    fontSize: "16px",
    cursor: "pointer",
    backgroundColor: "#FF8C00",
    color: "white",
    border: "none",
    borderRadius: "5px",
  },
  responseContainer: {
    marginTop: "20px",
  },
  responseText: {
    whiteSpace: "pre-wrap",
    wordWrap: "break-word",
    backgroundColor: "#FFF5EE",
    padding: "10px",
    borderRadius: "5px",
    display: "inline-block",
    maxWidth: "80%",
    textAlign: "left",
  },
  errorText: {
    color: "red",
    marginBottom: "10px",
  },
};

export default App;
