from ultralytics import YOLO
import streamlit as st
from PIL import Image

if __name__ == '__main__':

    st.set_page_config(layout="wide")

    myhtml = '''
        <div class="topnav" id="home">
            <div class= "logo">
                <a href="#">
                    <h3 style="font-size: 2.3rem;">Bubble-Detector</h3>
                </a>
            </div>
            <div class = "links">
                <a href="#speech-bubble-detector">Home</a>
                <a href="#speech-bubble-detection">Detector</a>
                <a href="#about">About</a>
                <a style="color: black; background: white; padding: 0 3em; display: flex; align-items: center" 
                href="https://github.com/AymanKUMA/yolov8_speech_bubbles_detection">Github</a>
            </div>
        </div>
        <div class="hidden">
        </div>
        <div class="content">
            <div class="home">
                <h1 id="home">Speech Bubble Detector</h1>
                <p>Revolutionize Manga and Comic Reading Experience with YOLOv8 
                </br>The Cutting-Edge Model That Detects Speech Bubbles with Unmatched Precision</p>
            </div>
        </div>
    '''

    style = '''
    <style>
    body {
        background-image: url("https://wallpapercave.com/wp/wp8578059.jpg");
        background-blend-mode: multiply;
        background-color: gray;
        background-size: cover;
    }
    
    .topnav {
        overflow: hidden;
        background-color: #333;
        padding: 0rem 1rem;
        display: flex;
        justify-content: space-between;
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 9999999999;
    }
    
    .logo a{
        text-decoration: none;
    }
    
    .links{
        width: 40%;
        display: flex;
        justify-content: space-around;
    }
    
    .links a{
        display: block;
        color: #f2f2f2;
        text-align: center;
        padding: 20px 10px;
        text-decoration: none;
    }
    
    .hidden{
        height: 5rem;
    }
    
    .content{
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .home{
        width: 80%;
        background-color: #00000080;
        padding: 60px;
    }
    
    .home  h1{
        font-size: 6rem;
    }
    
    .home p{
        font-size: 1.5rem;
    }
    
    main {
	    margin: 20px;
	    width: 80%;
        background-color: #00000080;
        padding: 60px;
    }
    
    .down{
        display: flex;
        align-items: center;
        justify-content: center;
    }

    section {
	    margin-bottom: 40px;
    }

    .gallery {
	    display: flex;
	    flex-wrap: wrap;
	    justify-content: center;
    }

    .gallery img {
	    margin: 10px;
	    max-width: 400px;
	    max-height: 400px;
    }
    
    .e19lei0e1{
        display: none;
    }
    
    .e8zbici2{
        display: none;
    }
    .e1tzin5v0{
        gap: 0rem;
    }
    .egzxvld4{
        padding: 0rem 0rem;
    }
    .egzxvld1{
        display: none;
    }
    .css-ffhzg2{
    background: none;
    }
    .css-keje6w{
            padding: 2rem 5rem;
            flex: 1 1 calc(100% - 1rem);
    }
    
    .e16nr0p32{
        display: none;
    }
    
    .footer{
        buttom: 0;
        width: 100%;
        padding: 1.3rem;
        text-align: center;
        color: white;
        background-color: #333;
        zindex: 9999999
    }
    </style>
    '''

    st.markdown(style, unsafe_allow_html=True)
    st.markdown(myhtml, unsafe_allow_html=True)
    col1, col2, col3 = st.columns((2, 1, 1))

    with col1:
        st.title('Speech bubble detection ')
        uploaded_file = st.file_uploader("Load image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Uploading image...'):
            with col2:
                st.image(uploaded_file, caption="Input Page", use_column_width=True)
                picture = Image.open(uploaded_file)
    else:
        is_valid = False
    if is_valid:
        with col3:
            with st.spinner(text='Processing image...'):
                model = YOLO('best.pt')
                results = model.predict(task="detect", source=picture, conf=0.85)
                img = results[0].plot()
            st.image(img, caption="Detected Objects", use_column_width=True)

    my2ndHtml = '''
    <div class="down">
        <main>
		    <section>
			    <h2>About</h2>
			    <p>Our model detects speech bubbles from manga and comics using YOLOv8 by ultralytics. With a custom dataset of 2000 images, our model is able to accurately detect and classify speech bubbles in a wide range of styles and formats.</p>
			    <p>Speech bubbles are an essential part of comic books and manga, allowing characters to speak and express emotions. Our model makes it easy to extract speech bubbles from images, making it a valuable tool for researchers, artists, and publishers alike.</p>
			    <p>This model is for academic use ONLY. Do not use it for any commercial purpose.</p>
		    </section>
		    <section>
			    <h2>Examples</h2>
			    <div class="gallery">
				    <img src=" https://drive.google.com/uc?id=1KJYsh3OX-WGAq5o_1P5k9ElvgOkt978w" alt="Example 1">
				    <img src="https://drive.google.com/uc?id=1fVvDcxzI46PTn0qRZwBTT6ReaVMNZIbI" alt="Example 2">
				    <img src="https://drive.google.com/uc?id=1pRBYx4P7_iTsNp3uV8ZeUQHvapWvBkO4" alt="Example 3">
				    <img src="https://drive.google.com/uc?id=1oxx4rrlHAa0nXU2vRXjaumghdN2czia-" alt="Example 4">
			    </div>
		    </section>
	    </main>
	</div>
    '''
    st.markdown(my2ndHtml, unsafe_allow_html=True)
    my3rdHtml = '''<div class="footer">&#169; Speech Bubble Detector</div>'''
    st.markdown(my3rdHtml, unsafe_allow_html=True)