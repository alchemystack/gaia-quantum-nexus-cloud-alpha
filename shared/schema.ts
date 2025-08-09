import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, real, timestamp, jsonb, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const quantumSessions = pgTable("quantum_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  prompt: text("prompt").notNull(),
  profile: varchar("profile", { length: 20 }).notNull().default('medium'),
  temperature: real("temperature").notNull().default(0.7),
  maxTokens: integer("max_tokens").notNull().default(128),
  generatedText: text("generated_text"),
  quantumInfluences: jsonb("quantum_influences").default('[]'),
  performanceMetrics: jsonb("performance_metrics"),
  entropyUsed: integer("entropy_used").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  completedAt: timestamp("completed_at"),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertQuantumSessionSchema = createInsertSchema(quantumSessions).omit({
  id: true,
  createdAt: true,
  completedAt: true,
}).extend({
  profile: z.enum(['strict', 'light', 'medium', 'spicy']),
  temperature: z.number().min(0.1).max(2.0),
  maxTokens: z.number().min(1).max(2000),
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertQuantumSession = z.infer<typeof insertQuantumSessionSchema>;
export type QuantumSession = typeof quantumSessions.$inferSelect;

// WebSocket message types
export const generationRequestSchema = z.object({
  prompt: z.string().min(1),
  profile: z.enum(['strict', 'light', 'medium', 'spicy']).default('medium'),
  maxTokens: z.number().min(1).max(2000).default(128),
  temperature: z.number().min(0.1).max(2.0).default(0.7),
});

export const tokenResponseSchema = z.object({
  token: z.string(),
  influence: z.string(),
  layerAnalysis: z.record(z.string(), z.number()),
  performanceMetrics: z.object({
    latency: z.number(),
    tokensPerSec: z.number(),
    entropyUsed: z.number(),
  }),
});

export type GenerationRequest = z.infer<typeof generationRequestSchema>;
export type TokenResponse = z.infer<typeof tokenResponseSchema>;
