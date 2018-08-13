from bs4 import BeautifulSoup as Soup

def add_image_tags(n):
    with open('./templates/result.html') as fp:
        html = Soup(fp, 'lxml')
        tag = html.find_all('div')[2]
        for i in range(n): 
            img = html.new_tag('img', id="mySlides", src="data:image/png;base64,{{{{ image_data[{number}] }}}}".format(number=i), width=500, height=500, style="width:100%")
            tag.append(img)
        #print(html.prettify())

if __name__ == "__main__":
    add_image_tags(4)