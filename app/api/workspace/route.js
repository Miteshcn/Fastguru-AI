import supabase from "@/lib/database";

export async function POST(request) {
  try {
    const body = await request.json();
    console.log("Received body:", body); 

    const { name, description, image, wallet_address } = body;

    if (!name || !description) {
      return new Response(
        JSON.stringify({ error: "Name and description are required." }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const { data, error } = await supabase
      .from("Workspace")
      .insert([{ name, description, image, wallet_address }])
      .select(); 

    if (error) {
      console.error("Supabase insert error:", error.message);
      throw error;
    }

    if (!data || data.length === 0) {
      throw new Error("Insert failed, no data returned.");
    }

    return new Response(
      JSON.stringify({ success: true, id: data[0].id }),
      { status: 201, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error creating workspace:", error.message);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}


export async function GET(request) {
  try {
    const url = new URL(request.url);
    const walletAddress = url.searchParams.get("wallet_address");

    let query = supabase.from("Workspace").select("*");

    if (walletAddress) {
      query = query.eq("wallet_address", walletAddress);
    }

    const { data, error } = await query;
    if (error) throw error;

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error fetching workspaces:", error.message);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}


export async function DELETE(request) {
  try {
    // const { id } = await request.json();

    const url = new URL(request.url);
    const id = url.searchParams.get("id");

    const { error } = await supabase
      .from("Workspace")
      .delete()
      .eq("id", id);

    if (error) throw error;

    return new Response(
      JSON.stringify({ success: true }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error deleting workspace:", error.message);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}