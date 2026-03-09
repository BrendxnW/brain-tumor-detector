import { useState } from "react"
import './App.css'

function App() {
  const [preview, setPreview] = useState("")
  
  const handleFileChnage = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const imageUrl = URL.createObjectURL(file)
      setPreview(imageUrl)

    }
  }

  return (
    <div className="upload-container">
      <h1 className='title'>Brain Tumor Detector</h1>
      
        {preview && <img src={preview} alt="MRI preview" width="800" />}
        
        <input 
        type="file"
        accept="image/jpg, image/jpeg, image/png"
        onChange={handleFileChnage}
        />
        
        <button>Submit MRI</button>
      
    </div>
  

  )
}

export default App
