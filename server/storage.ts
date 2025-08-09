import { type User, type InsertUser, type QuantumSession, type InsertQuantumSession } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Quantum session methods
  createQuantumSession(session: InsertQuantumSession): Promise<QuantumSession>;
  getQuantumSession(id: string): Promise<QuantumSession | undefined>;
  updateQuantumSession(id: string, updates: Partial<QuantumSession>): Promise<QuantumSession | undefined>;
  getUserSessions(userId: string): Promise<QuantumSession[]>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private sessions: Map<string, QuantumSession>;

  constructor() {
    this.users = new Map();
    this.sessions = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createQuantumSession(insertSession: InsertQuantumSession): Promise<QuantumSession> {
    const id = randomUUID();
    const session: QuantumSession = {
      ...insertSession,
      id,
      generatedText: null,
      quantumInfluences: [],
      performanceMetrics: null,
      entropyUsed: 0,
      createdAt: new Date(),
      completedAt: null,
    };
    this.sessions.set(id, session);
    return session;
  }

  async getQuantumSession(id: string): Promise<QuantumSession | undefined> {
    return this.sessions.get(id);
  }

  async updateQuantumSession(id: string, updates: Partial<QuantumSession>): Promise<QuantumSession | undefined> {
    const session = this.sessions.get(id);
    if (!session) return undefined;
    
    const updatedSession = { ...session, ...updates };
    this.sessions.set(id, updatedSession);
    return updatedSession;
  }

  async getUserSessions(userId: string): Promise<QuantumSession[]> {
    return Array.from(this.sessions.values()).filter(
      (session) => session.userId === userId
    );
  }
}

export const storage = new MemStorage();
