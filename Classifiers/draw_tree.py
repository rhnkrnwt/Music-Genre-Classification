from PIL import Image,ImageDraw
from decision_tree import *

def getwidth(tree):
  if tree.tb==None and tree.fb==None: return 1
  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
  if tree.tb==None and tree.fb==None: return 0
  return max(getdepth(tree.tb),getdepth(tree.fb))+1

def drawtree(tree,jpeg='dt.jpg'):
  w=getwidth(tree)*50
  h=getdepth(tree)*100+120

  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  drawnode(draw,tree,w/2,20)
  img.save(jpeg,'JPEG')

def drawnode(draw,tree,x,y):
  if tree.result==None:
    # Get the width of each branch
    w1=getwidth(tree.fb)*50
    w2=getwidth(tree.tb)*50

    # Determine the total space required by this node
    left=x-(w1+w2)/2
    right=x+(w1+w2)/2

    # Draw the condition string
    draw.text((x-20,y-10),str(tree.feature)+':'+str(tree.val),(0,0,0))

    # Draw links to the branches
    draw.line((x,y,left+w1/2,y+100),fill=(0,0,255))
    draw.line((x,y,right-w2/2,y+100),fill=(0,0,255))

    # Draw the branch nodes
    drawnode(draw,tree.fb,left+w1/2,y+100)
    drawnode(draw,tree.tb,right-w2/2,y+100)
  else:
    txt=' \n'.join(['%s:%d'%v for v in tree.result.items()])
    draw.text((x-20,y),txt,(0,0,0))

tree = None
with open('forest/d_tree0.pickle', 'rb') as f:
    tree = pickle.load(f)
drawtree(tree)
