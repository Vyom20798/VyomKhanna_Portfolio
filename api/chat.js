import { OpenAIApi, Configuration } from 'openai';
import { config } from 'dotenv';
import cors from 'cors';
import express from 'express';

config();

const app = express();
app.use(cors({ origin: '*' }));
app.use(express.json());

const openai = new OpenAIApi(new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
}));

app.post('/api/chat', async (req, res) => {
  const { message } = req.body;

  try {
    const completion = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: message }],
    });

    res.json({ reply: completion.data.choices[0].message.content });
  } catch (error) {
    console.error(error.response ? error.response.data : error.message);
    res.status(500).json({ reply: 'Sorry, something went wrong. Please try again later.' });
  }
});

// ✨ Export it as serverless handler
export default app;
