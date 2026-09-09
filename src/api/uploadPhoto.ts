export async function uploadPhoto(
  file: File,
): Promise<{ success: boolean; message: string }> {
  const formData = new FormData();
  formData.append('photo', file);
  const response = await fetch('/api/upload', { method: 'POST', body: formData });
  if (!response.ok) throw new Error('Upload failed');
  return response.json();
}
