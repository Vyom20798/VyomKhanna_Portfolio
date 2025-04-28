import { Configuration, OpenAIApi } from 'openai';
import { config } from 'dotenv';

// Load your environment variables (like your OPENAI_API_KEY)
config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

// Serverless function for Vercel
export default async function handler(req, res) {
  // Set CORS Headers manually
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*'); // Allow all origins for now
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { message } = req.body;

  try {
    const completion = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: message }],
    });

    const reply = completion.data.choices[0].message.content;
    res.status(200).json({ reply });
  } catch (error) {
    console.error('OpenAI API Error:', error.response ? error.response.data : error.message);
    res.status(500).json({ error: 'Failed to get response from OpenAI' });
  }
}
