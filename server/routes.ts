import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { llmEngine } from "./services/llm-engine";
import { ModalLLMEngine } from "./services/modal-llm-engine";
import { QuantumBlockchainsQRNG } from "./services/qrng";
import { influenceTranslator } from "./services/influence-translator";
import { generationRequestSchema, type TokenResponse } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // Initialize the appropriate LLM engine based on configuration
  const qrng = new QuantumBlockchainsQRNG();
  const modalEngine = new ModalLLMEngine(qrng);
  const isModalConfigured = await modalEngine.isConfigured();
  
  // Use Modal-hosted GPT-OSS 120B if configured, otherwise use local demo
  const activeEngine = isModalConfigured ? modalEngine : llmEngine;
  
  if (isModalConfigured) {
    console.log('[Routes] Using Modal-hosted GPT-OSS 120B model with QRNG logit modification');
  } else {
    console.log('[Routes] Modal not configured. Set MODAL_ENDPOINT and MODAL_API_KEY to use the real 120B model');
  }

  // REST API routes
  app.get("/api/qrng-status", async (req, res) => {
    try {
      const status = await activeEngine.getQRNGStatus();
      res.json({
        ...status,
        modelEngine: isModalConfigured ? 'Modal GPT-OSS 120B' : 'Local Demo'
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to get QRNG status" });
    }
  });

  app.get("/api/sessions/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const sessions = await storage.getUserSessions(userId);
      res.json(sessions);
    } catch (error) {
      res.status(500).json({ error: "Failed to get sessions" });
    }
  });

  app.post("/api/sessions", async (req, res) => {
    try {
      const sessionData = generationRequestSchema.parse(req.body);
      const session = await storage.createQuantumSession({
        ...sessionData,
        userId: null // For demo purposes, no user auth
      });
      res.json(session);
    } catch (error) {
      res.status(400).json({ error: "Invalid session data" });
    }
  });

  // HTTP fallback endpoint for development mode
  app.post("/api/generate", async (req, res) => {
    console.log('[API] Generate request received:', { 
      method: req.method,
      body: req.body 
    });
    
    try {
      // Validate request
      if (!req.body) {
        res.status(400).json({ error: 'No request body provided' });
        return;
      }
      
      const request = generationRequestSchema.parse(req.body);
      console.log('[API] Validated request:', request);
      
      // Set up Server-Sent Events for streaming
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'X-Accel-Buffering': 'no' // Disable nginx buffering
      });

      const sessionId = `dev-${Date.now()}`;
      let tokenCount = 0;

      // Generate tokens using the active engine (Modal if configured, local otherwise)
      console.log('[API] Starting generation with', isModalConfigured ? 'Modal GPT-OSS 120B' : 'local demo', 'engine...');
      const generator = activeEngine.generate(
        request.prompt,
        request.profile,
        request.maxTokens,
        request.temperature
      );
      
      for await (const tokenResponse of generator) {
        tokenCount++;
        const message = JSON.stringify({
          type: 'token',
          ...tokenResponse
        });
        res.write(`data: ${message}\n\n`);
        
        // Log first few tokens for debugging
        if (tokenCount <= 3) {
          console.log(`[API] Sent token ${tokenCount}:`, tokenResponse.token);
        }
      }

      // Send completion event
      console.log(`[API] Generation complete. Total tokens: ${tokenCount}`);
      res.write(`data: ${JSON.stringify({
        type: 'complete',
        sessionId,
        totalTokens: tokenCount
      })}\n\n`);

      res.end();
    } catch (error) {
      console.error('[API] Generation error:', error);
      
      // If headers not sent yet, send error response
      if (!res.headersSent) {
        res.status(500).json({ 
          error: error instanceof Error ? error.message : 'Generation failed' 
        });
      } else {
        // If streaming already started, send error event
        res.write(`data: ${JSON.stringify({
          type: 'error',
          error: error instanceof Error ? error.message : 'Generation failed'
        })}\n\n`);
        res.end();
      }
    }
  });
  
  // Handle incorrect GET requests to generate endpoint
  app.get("/api/generate", (req, res) => {
    res.status(405).json({ 
      error: 'Method not allowed. Use POST to generate text.' 
    });
  });

  // WebSocket server for real-time text generation
  const wss = new WebSocketServer({ 
    server: httpServer, 
    path: '/ws/generate'
  });

  wss.on('connection', (ws: WebSocket) => {
    console.log('WebSocket client connected');

    ws.on('message', async (data: Buffer) => {
      try {
        const request = JSON.parse(data.toString());
        const validatedRequest = generationRequestSchema.parse(request);
        
        console.log('Starting generation with profile:', validatedRequest.profile);

        // Create session in storage
        const session = await storage.createQuantumSession({
          ...validatedRequest,
          userId: null
        });

        // Stream generation using active engine (Modal if configured, local otherwise)
        const stream = activeEngine.generate(
          validatedRequest.prompt,
          validatedRequest.profile,
          validatedRequest.maxTokens,
          validatedRequest.temperature
        );
        let generatedTokens: string[] = [];
        let allInfluences: string[] = [];

        for await (const response of stream) {
          if (ws.readyState === WebSocket.OPEN) {
            generatedTokens.push(response.token);
            allInfluences.push(response.influence);
            
            // Send token response
            ws.send(JSON.stringify(response));
          } else {
            break;
          }
        }

        // Update session with final results
        if (session) {
          await storage.updateQuantumSession(session.id, {
            generatedText: generatedTokens.join(' '),
            quantumInfluences: allInfluences,
            completedAt: new Date(),
            entropyUsed: generatedTokens.length * 12 // Approximate entropy usage
          });
        }

        // Send completion signal
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ 
            type: 'complete',
            sessionId: session?.id,
            totalTokens: generatedTokens.length
          }));
        }

      } catch (error) {
        console.error('Generation error:', error);
        if (ws.readyState === WebSocket.OPEN) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          let userMessage = errorMessage;
          
          // Provide user-friendly error messages for QRNG failures
          if (errorMessage.includes('QRNG API key not available')) {
            userMessage = 'True quantum randomness is required but no QRNG API key is configured. Please contact administrator to set up Quantum Blockchains API access.';
          } else if (errorMessage.includes('No quantum data received from QRNG API')) {
            userMessage = 'Unable to obtain true quantum randomness from QRNG service. Generation stopped to maintain quantum authenticity.';
          } else if (errorMessage.includes('QRNG API Error') || errorMessage.includes('request timeout')) {
            userMessage = 'Quantum Blockchains API is temporarily unavailable. True quantum randomness cannot be obtained at this time.';
          }
          
          ws.send(JSON.stringify({ 
            type: 'error', 
            message: userMessage 
          }));
        }
      }
    });

    ws.on('close', () => {
      console.log('WebSocket client disconnected');
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  });

  return httpServer;
}
