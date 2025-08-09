# Gaia Quantum Nexus Cloud

## Overview

Gaia Quantum Nexus Cloud is a quantum-augmented language model interface that integrates true quantum randomness from Quantum Blockchains QRNG API into text generation. **CRITICAL: This system uses ONLY true quantum randomness with NO fallback to pseudorandomness. Generation will HALT if QRNG is unavailable.** The application provides a sophisticated web interface for experimenting with quantum-influenced AI text generation, featuring real-time performance metrics, layer analysis, and visual quantum effects. The system combines React frontend with Express backend, using HTTP streaming for real-time token generation and PostgreSQL for session persistence.

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
- **NO FALLBACK**: System halts if QRNG unavailable (strict quantum-only policy)
- **Dual Output**: Shows both QRNG-modified text and vector interpretation
- **Layer Analysis**: Real-time monitoring of attention, FFN, and embedding layers

### Cloud Model Deployment (GPT-OSS 120B GGUF)
- **Model**: ggml-org/gpt-oss-120b-GGUF from HuggingFace
- **Model Link**: https://huggingface.co/ggml-org/gpt-oss-120b-GGUF
- **Running Command**: `llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 -fa --jinja --reasoning-format none`
- **GPU Requirements**: 1x A100 with 64GB RAM
- **Deployment Scripts**: Modal serverless deployment in `MODAL_WEB_NOTEBOOK.py`
- **Integration**: ModalLLMEngine in `server/services/modal-llm-engine.ts`
- **Low-Latency Setup**: Model stays loaded (keep_warm=1), Flash Attention enabled
- **Cost Optimization**: Modal at ~$60/month for light usage with 1 GPU vs 2 GPUs

### Authentication & Authorization
- **Current State**: No authentication implemented (demo mode)
- **Planned**: User sessions with PostgreSQL storage
- **Session Storage**: Connect-pg-simple for production session management

## External Dependencies

### Core Services
- **Quantum Blockchains QRNG API**: Primary source of quantum randomness for text generation
- **Neon Database**: Serverless PostgreSQL hosting
- **Replit**: Development and deployment platform

### Cloud Model Hosting (GPT-OSS 120B)
- **Modal (Recommended)**: Serverless GPU hosting, ~$95/month for light usage, 10-30s cold start
- **RunPod**: Dedicated A100 80GB instances, $2-3/hour, lowest latency option
- **AWS SageMaker**: Enterprise-grade hosting, P4d.24xlarge instances, ~$24k/month for 24/7
- **Replicate**: Easy deployment, $0.001/sec GPU time, higher latency
- **Together AI**: Managed service option (requires custom arrangement)

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