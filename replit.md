# Gaia Quantum Nexus Cloud

## Overview

Gaia Quantum Nexus Cloud is a quantum-augmented language model interface that integrates true quantum randomness from Quantum Blockchains QRNG API into text generation. The application provides a sophisticated web interface for experimenting with quantum-influenced AI text generation, featuring real-time performance metrics, layer analysis, and visual quantum effects. The system combines React frontend with Express backend, using WebSocket for real-time token streaming and PostgreSQL for session persistence.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **UI Components**: Shadcn/ui component library with Radix UI primitives
- **Styling**: Tailwind CSS with a dark quantum-themed design system
- **State Management**: React hooks with TanStack Query for server state
- **Real-time Communication**: Custom WebSocket hook for streaming token generation
- **Routing**: Wouter for lightweight client-side routing

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **WebSocket**: ws library for real-time token streaming
- **Database ORM**: Drizzle ORM with PostgreSQL dialect
- **Session Storage**: In-memory storage with planned database integration
- **QRNG Integration**: Custom quantum random number generator service using Quantum Blockchains API

### Data Storage
- **Database**: PostgreSQL with Neon serverless driver
- **ORM**: Drizzle ORM for type-safe database operations
- **Schema**: Users and quantum sessions tables with JSON fields for complex data
- **Migrations**: Drizzle Kit for schema management and migrations
- **Session Management**: Planned integration with connect-pg-simple for persistent sessions

### Quantum Integration
- **QRNG Provider**: Quantum Blockchains API for true quantum randomness
- **Entropy Pool**: Buffered quantum random data for improved performance
- **Fallback**: Crypto-secure randomness when quantum source unavailable
- **Influence Translation**: Custom service to translate quantum effects into meaningful descriptions
- **Layer Analysis**: Real-time monitoring of attention, FFN, and embedding layers

### Authentication & Authorization
- **Current State**: No authentication implemented (demo mode)
- **Planned**: User sessions with PostgreSQL storage
- **Session Storage**: Connect-pg-simple for production session management

## External Dependencies

### Core Services
- **Quantum Blockchains QRNG API**: Primary source of quantum randomness for text generation
- **Neon Database**: Serverless PostgreSQL hosting
- **Replit**: Development and deployment platform

### UI & Styling
- **Shadcn/ui**: Pre-built accessible UI components
- **Radix UI**: Headless component primitives
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Icon library
- **Google Fonts**: Inter font family

### Development Tools
- **Vite**: Frontend build tool and dev server
- **TypeScript**: Type safety across the stack
- **ESBuild**: Production backend bundling
- **Drizzle Kit**: Database schema management
- **TanStack Query**: Server state management
- **React Hook Form**: Form validation and management

### Runtime Dependencies
- **Express.js**: Web application framework
- **WebSocket (ws)**: Real-time communication
- **Drizzle ORM**: Database operations
- **Zod**: Runtime type validation
- **Date-fns**: Date manipulation utilities
- **Class Variance Authority**: Component variant management