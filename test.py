import requests
import json

# Replace with the actual URL of your Flask app
# If running in Colab with ngrok, this would be the ngrok URL
# If running locally, it would likely be 'http://127.0.0.1:5000/predict'
url = 'http://127.0.0.1:5000/predict' # Replace with your Flask app URL

# Prepare the new data in a dictionary format that matches your original features
# This is an example; replace with your actual new data
new_data = {
    'project_type': ['Perumahan'],
    'client_type': ['BUMN'],
    'contract_type': ['Lump Sum'],
    'is_design_and_build': [1],
    'nilai_kontrak_awal_miliar_rp': [70.00],
    'total_jam_kerja_estimasi': [280000],
    'volume_pekerjaan_tanah_m3': [35000],
    'volume_beton_m3': [18000],
    'berat_baja_struktural_ton': [0],
    'panjang_instalasi_utama_km': [0],
    'jumlah_titik_akhir_instalasi': [0],
    'jumlah_item_pekerjaan_utama': [50],
    'tingkat_risiko_geoteknik': [2],
    'lokasi_provinsi': ['Jawa Barat'],
    'lokasi_urban_rural': ['Urban'],
    'musim_pelaksanaan': ['Kemarau'],
    'indeks_harga_komoditas_saat_mulai': [105.00],
    'jumlah_kompetitor_saat_tender': [5],
    'pengalaman_pm_tahun': [20],
    'jumlah_sdm_inti': [15],
    'persentase_subkontraktor': [45]
}

# Convert the data to JSON format
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(new_data), headers=headers)

# Print the predictions
if response.status_code == 200:
    predictions = response.json()
    print("Predictions:")
    print(predictions)
else:
    print(f"Error: {response.status_code}")
    print(response.text)