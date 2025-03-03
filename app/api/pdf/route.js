import supabase from "@/lib/database";

export async function POST(request) {
  try {
    const body = await request.json();
    const { workspace_id, file_url, fileName } = body;

    if (!workspace_id || !file_url || !fileName) {
      return new Response(
        JSON.stringify({
          error: "Workspace ID, file URL, and file name are required.",
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

   const { data, error } = await supabase
  .from("PDF")
  .insert([
    {
      workspace_id,
      file_url,
      fileName,
    },
  ])
  .select();

      console.log("supabase data", data);

    if (error) {
      console.error("Error uploading PDF:", error.message);
      throw error;
    }

    return new Response(
      JSON.stringify({ success: true, id: data[0]?.id }),
      {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error uploading PDF:", error.message);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}

export async function GET(request) {
  try {
    const url = new URL(request.url);
    const workspaceId = url.searchParams.get("workspace_id");

    if (!workspaceId) {
      return new Response(
        JSON.stringify({ error: "Workspace ID is required." }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const { data: pdfs, error } = await supabase
      .from("PDF")
      .select("*")
      .eq("workspace_id", workspaceId);

    if (error) {
      console.error("Error fetching PDFs:", error.message);
      throw error;
    }

    return new Response(JSON.stringify(pdfs), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error fetching PDFs:", error.message);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
