import axios from "axios";
import FormData from "form-data";

const PINATA_API_KEY = "737602a1d514ee97e097";
const PINATA_API_SECRET = "bb8d6e3d250ef5a6d7959cfaec1cd0e98dee0c9c92e0d958195fe59522aa5833";

export async function POST(req) {
  try {
    const { file, fileName } = await req.json(); 

    if (!file || !fileName) {
      return new Response(
        JSON.stringify({ error: "File and fileName are required." }),
        { status: 400 }
      );
    }

    // Prepare the file for Pinata
    const formData = new FormData();
    formData.append("file", Buffer.from(file, "base64"), fileName);

    // Upload to Pinata
    const response = await axios.post(
      "https://api.pinata.cloud/pinning/pinFileToIPFS",
      formData,
      {
        headers: {
          ...formData.getHeaders(),
                    pinata_api_key: PINATA_API_KEY,
          pinata_secret_api_key: PINATA_API_SECRET,
        },
      }
    );

    const { IpfsHash } = response.data;
    return new Response(
      JSON.stringify({ message: "File uploaded successfully", hash: IpfsHash }),
      { status: 200 }
    );
  } catch (error) {
    console.error("Error uploading file:", error);
    return new Response(
      JSON.stringify({ error: "File upload failed" }),
      { status: 500 }
    );
  }
}

