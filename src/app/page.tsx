"use client";

import { useState } from "react";
import { useChat } from "ai/react";

export default function Chat() {
	const { messages, input, handleInputChange, handleSubmit } = useChat({
		api: "/api/chat",
	});

	return (
		<div className="flex flex-col w-full max-w-xl mx-auto h-screen p-4">
			<div className="flex-1 overflow-y-auto space-y-4">
				{messages.map((message) => (
					<div
						key={message.id}
						className={`flex ${
							message.role === "assistant" ? "justify-start" : "justify-end"
						}`}
					>
						<div
							className={`rounded-lg px-4 py-2 max-w-[80%] ${
								message.role === "assistant"
									? "bg-blue-500 text-white"
									: "bg-gray-200"
							}`}
						>
							{message.content}
						</div>
					</div>
				))}
			</div>

			<form onSubmit={handleSubmit} className="flex space-x-4 pt-4">
				<input
					className="flex-1 rounded-lg border border-gray-300 p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
					value={input}
					onChange={handleInputChange}
					placeholder="Say something to Riley..."
				/>
				<button
					type="submit"
					className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
				>
					Send
				</button>
			</form>
		</div>
	);
}
