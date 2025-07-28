import { useState } from "react";
import { Button } from "./components/ui/button";
import { Loader2, Sparkle } from "lucide-react";
import { Card, CardContent } from "./components/ui/card";
import { Label } from "./components/ui/label";
import { Input } from "./components/ui/input";
import { Badge } from "./components/ui/badge";
import Markdown from 'react-markdown'

type Metadata = {
  image_id: string;
  image_path: string;
  caption: string;
  tags: string[];
  location: string;
};

type Response = {
  response: string;
  sources: Metadata[];
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [k, setK] = useState(3);
  const [response, setResponse] = useState<Response | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const reset = () => {
    setQuestion("");
    setK(3);
    setResponse(null);
    setLoading(false);
    setError("");
  };

  const handleAsk = async () => {
    reset();

    if (!question || question.trim() === "") {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    try {
      const api = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question, k }),
      });
      const response = await api.json();
      setResponse(response);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-muted px-6 py-12 flex flex-col items-center">
      <div className="max-w-4xl w-full space-y-6">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-primary">
            AI for Sustainable Community
          </h1>
          <p className="text-muted-foreground text-lg">
            Analyze urban scenes using AI to support Sustainable Development
            Goals (SDGs).
          </p>
        </header>

        <Card>
          <CardContent className="flex flex-col gap-4">
            <Label>What would you like to ask?</Label>
            <Input
              type="text"
              placeholder="e.g., What areas show signs of poor infrastructure?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />

            <div className="flex items-center gap-4">
              <label className="text-gray-700 font-medium">
                Top-K Results:
              </label>
              <input
                type="number"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-20 border border-gray-300 rounded px-2 py-1"
                min={1}
                max={10}
              />
            </div>

            <Button onClick={handleAsk} disabled={loading}>
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="animate-spin" />
                  <span>Loading...</span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Sparkle />
                  <span>Ask</span>
                </span>
              )}
            </Button>
          </CardContent>
        </Card>

        {error && <p className="text-destructive text-center">{error}</p>}

        {response && (
          <Card>
            <CardContent className="">
              <div className="space-y-4">
                <h2 className="text-xl font-semibold mb-2">Response:</h2>
                <Markdown>
                  {response.response}
                </Markdown>
              </div>
              <div className="space-y-4 pt-4">
                <h2 className="text-xl font-semibold mb-2">
                  Sources ({response.sources.length}):
                </h2>
                <div className="grid gap-4">
                  {response.sources.map((source) => {
                    const imagePath = `http://localhost:5000/images/${source.image_path
                      .split("/")
                      .pop()}`;
                    return (
                      <div className="flex gap-4 rounded-lg border p-4">
                        <img
                          src={imagePath}
                          width={300}
                          height={300}
                          className="rounded-lg"
                        />

                        <div className="">
                          <p className="text-sm mb-2">{source.caption}</p>
                          {source.tags.map((tag) => (
                            <Badge variant="outline" className="capitalize">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
