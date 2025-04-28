import { Configuration, OpenAIApi } from "openai";
import { config } from "dotenv";

config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY, // Safe way — from environment variables
});

const openai = new OpenAIApi(configuration);

export default async function handler(req, res) {
  // ✅ Proper CORS setup
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*'); // For now allow all, can later specify
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin');

  if (req.method === 'OPTIONS') {
    return res.status(200).end(); // Handle pre-flight check
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  const { message } = req.body;

  try {
    const completion = await openai.createChatCompletion({
      model: "gpt-3.5-turbo", // If you use gpt-4, change it here
      messages: [{ role: "user", content: message }],
    });

    const reply = completion.data.choices[0].message.content;
    res.status(200).json({ reply });
  } catch (error) {
    console.error(error.response ? error.response.data : error.message);
    res.status(500).json({ error: "Failed to get response from OpenAI" });
  }
}
