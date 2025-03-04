import { ChatOpenAI } from "langchain/chat_models";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { OpenAI } from "langchain/llms";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { ConversationBufferMemory } from "langchain/memory";
import supabase from "@/lib/database";
import fetch from "node-fetch";
import { PDFDocument, rgb } from "pdf-lib";

const API_KEY = process.env.NEXT_PUBLIC_OPENAI_API_KEY;

export async function POST(request) {
  try {
    const workspaceId = request.headers.get("workspace_id");
    const isRealEstateAgent = request.headers.get("isRealEstateAgent") === "true";

    if (!workspaceId) {
      return new Response(
        JSON.stringify({ error: "Workspace ID is required." }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const body = await request.json();
    const message = body.message;

    if (!message) {
      return new Response(
        JSON.stringify({ error: "No message provided." }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const { error: userInsertError } = await supabase
      .from("Chat")
      .insert({ workspace_id: workspaceId, content: message, role: "user" });

    if (userInsertError) {
      console.error("Error inserting user message:", userInsertError.message);
      throw userInsertError;
    }

    let assistantMessage = "";

    const memory = new ConversationBufferMemory({
      returnMessages: true,
      memoryKey: "chat_history",
    });

    if (isRealEstateAgent) {
      const embeddings = new OpenAIEmbeddings({ openAIApiKey: API_KEY });
      const vectorStore = new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: "vector_store",
        queryName: "match_vectors",
      });

      const retriever = vectorStore.asRetriever();
      const chain = ConversationalRetrievalQAChain.fromLLM(
        new OpenAI({ openAIApiKey: API_KEY }),
        retriever,
        { memory }
      );

      const response = await chain.call({ question: message });
      assistantMessage = response.text;
    } else {
      const chat = new ChatOpenAI({ openAIApiKey: API_KEY });
      assistantMessage = await chat.call(message, { memory });
    }
    const keywords = ["documentize", "document", "agreement"];
    const isDocumentRequest = keywords.some((keyword) =>
      message.toLowerCase().includes(keyword)
    );

    const storedMessage = isDocumentRequest
      ? "Check your Web3 Storage for the requested document."
      : assistantMessage;

    const { error: assistantInsertError } = await supabase
      .from("Chat")
      .insert({ workspace_id: workspaceId, content: storedMessage, role: "assistant" });

    if (assistantInsertError) {
      console.error("Error inserting assistant message:", assistantInsertError.message);
      throw assistantInsertError;
    }

    if (isDocumentRequest) {
      const pdfDoc = await PDFDocument.create();
      const page = pdfDoc.addPage();
      const { width, height } = page.getSize();
      page.drawText(assistantMessage, { x: 50, y: height - 50, size: 12, color: rgb(0, 0, 0) });

      const pdfBytes = await pdfDoc.save();
      return new Response(pdfBytes, {
        status: 200,
        headers: {
          "Content-Type": "application/pdf",
          "Content-Disposition": "attachment; filename=document.pdf",
        },
      });
    }

    return new Response(
      JSON.stringify({ reply: assistantMessage }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error:", error.message);
    return new Response(
      JSON.stringify({ error: "Failed to process request." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}

export async function GET(request) {
  try {
    const workspaceId = request.headers.get("workspace_id");

    if (!workspaceId) {
      return new Response(
        JSON.stringify({ error: "Workspace ID is required." }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const { data: chats, error } = await supabase
      .from("Chat")
      .select("*")
      .eq("workspace_id", workspaceId)
      .order("created_at", { ascending: true });

    if (error) {
      console.error("Error fetching chat history:", error.message);
      throw error;
    }

    return new Response(
      JSON.stringify({ chats }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error fetching chat history:", error.message);
    return new Response(
      JSON.stringify({ error: "Failed to fetch chat history." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
