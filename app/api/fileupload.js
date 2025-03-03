
import { PinataSDK } from "pinata";
const JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJmMzNlNWZhYy1jMjIwLTQ0YTAtODE5OC1jODg5MThmM2NlMjUiLCJlbWFpbCI6Im1haGVzd2FyMTQxMjAzQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJwaW5fcG9saWN5Ijp7InJlZ2lvbnMiOlt7ImRlc2lyZWRSZXBsaWNhdGlvbkNvdW50IjoxLCJpZCI6IkZSQTEifSx7ImRlc2lyZWRSZXBsaWNhdGlvbkNvdW50IjoxLCJpZCI6Ik5ZQzEifV0sInZlcnNpb24iOjF9LCJtZmFfZW5hYmxlZCI6ZmFsc2UsInN0YXR1cyI6IkFDVElWRSJ9LCJhdXRoZW50aWNhdGlvblR5cGUiOiJzY29wZWRLZXkiLCJzY29wZWRLZXlLZXkiOiI3NDcxZmViYzg3NGVhMzNhMjRmOCIsInNjb3BlZEtleVNlY3JldCI6IjFhMDg3YjVlM2RlM2E1MzQ0ZmMwMDEyYjdhMWZiYzE5ZjA3NGEyMzZlYTNjYmNlZGUzMzIxZTY0M2Y5YTg2MzgiLCJleHAiOjE3NTUzNzA2MjV9.rfLyawWTaubeYPnobWfQhsqQjhvz69IwPXLoUktR_Ug";

const uploadFileIPFS = async (uploadData) => {
  const pinata = new PinataSDK({
    pinataJwt: JWT,
    pinataGateway: "amethyst-hurt-catfish-974.mypinata.cloud",
  });

  const upload = await pinata.upload.file(uploadData);
  return (upload)
}


export default uploadFileIPFS;