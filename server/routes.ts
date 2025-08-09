import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { llmEngine } from "./services/llm-engine";
import { influenceTranslator } from "./services/influence-translator";
import { generationRequestSchema, type TokenResponse } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);

  // REST API routes
  app.get("/api/qrng-status", async (req, res) => {
    try {
      const status = await llmEngine.getQRNGStatus();
      res.json(status);
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

        // Stream generation
        const stream = llmEngine.generateStream(validatedRequest);
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
          ws.send(JSON.stringify({ 
            type: 'error', 
            message: error instanceof Error ? error.message : 'Unknown error' 
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
