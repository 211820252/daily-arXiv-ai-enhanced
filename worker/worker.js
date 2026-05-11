/**
 * Cloudflare Worker — LLM API 代理
 *
 * 部署步骤：
 *   1. npm install -g wrangler
 *   2. wrangler login
 *   3. wrangler secret put OPENAI_API_KEY   # 输入你的 API Key
 *   4. wrangler secret put OPENAI_BASE_URL  # 可选，默认 https://api.deepseek.com
 *   5. wrangler secret put MODEL_NAME       # 可选，默认 deepseek-v4-pro
 *   6. npx wrangler deploy
 *
 * 部署后会得到一个 URL（如 https://arxiv-summary.xxx.workers.dev），
 * 填入 Settings 页面的 "Full Summary API Endpoint" 即可。
 */

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    if (request.method !== 'POST') {
      return new Response('Method Not Allowed', {
        status: 405,
        headers: { 'Access-Control-Allow-Origin': '*' },
      });
    }

    try {
      const body = await request.json();
      const { content, prompt } = body;

      if (!content || !prompt) {
        return new Response(JSON.stringify({ error: 'Missing content or prompt' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
      }

      const baseUrl = (env.OPENAI_BASE_URL || 'https://api.deepseek.com').replace(/\/+$/, '');
      const modelName = env.MODEL_NAME || 'deepseek-v4-pro';

      const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: modelName,
          messages: [
            {
              role: 'system',
              content: '你是一个专业学术助手，请用中文回答。请严格基于提供的论文内容进行总结，不要编造信息。',
            },
            { role: 'user', content: prompt },
          ],
          max_tokens: 4096,
          temperature: 0.3,
        }),
      });

      if (!resp.ok) {
        const errText = await resp.text().catch(() => '');
        return new Response(
          JSON.stringify({ error: `LLM API error ${resp.status}: ${errText.slice(0, 300)}` }),
          {
            status: 502,
            headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
          }
        );
      }

      const data = await resp.json();
      const summary = data.choices?.[0]?.message?.content || '模型没有返回可读总结。';

      return new Response(JSON.stringify({ summary }), {
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      });
    } catch (err) {
      return new Response(
        JSON.stringify({ error: `Internal error: ${err.message}` }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        }
      );
    }
  },
};
