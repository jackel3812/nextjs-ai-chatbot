import { NextResponse } from "next/server";
import { RileyAI } from "@/lib/ai/core";

// Initialize Riley AI
const riley = new RileyAI();

export async function POST(req: Request) {
	try {
		const { messages } = await req.json();
		const lastMessage = messages[messages.length - 1];

		// Process through Riley's AI system
		const response = await riley.think(lastMessage.content);

		// Trigger self-improvement occasionally
		if (Math.random() < 0.1) {
			riley.selfImprove().catch(console.error);
		}

		return NextResponse.json({ response });
	} catch (error) {
		console.error("Error in chat route:", error);
		return NextResponse.json(
			{ error: "Internal server error" },
			{ status: 500 },
		);
	}
}
