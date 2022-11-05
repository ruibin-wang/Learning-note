# crewler methods note

Here we introduce ** kind of crewler methods, including: Beatifulsoup, Re,  Scrapy and Crawley

**beautifulsoup is more handable than re!!**


## Beautifulsoup

The turtorial can be found at: https://cloud.tencent.com/developer/article/1784030

* install 
    ```python
    pip install beautifulsoup4
    pip install requests

    # import
    from bs4 import Beautifulsoup
    import requests
    ```


* function
    ```python
    ## get all content of the web, as a format of text
    html = requests.get(url, headers=headers).text
    ## supported parser have: "lxml", "html5lib", "html.parser"
    soup = Beautifulsoup(html,parser)
    ```


* basic element

    ```
    Tag, 标签，最基本的信息组织单元，分别用<>和标明开头和结尾；
    Name, 标签的名字，<p>…</p>的名字是'p'，格式：<tag>.name;
    Attributes, 标签的属性，字典形式组织，格式：<tag>.attrs;
    NavigableString, 标签内非属性字符串，<>…</>中字符串，格式：<tag>.string;
    Comment, 标签内字符串的注释部分，一种特殊的NavigableString 对象类型;
    ```

    ```python
    ## .prettify() can be used for label, to make the output more elegent.
    ## how to use 
    <tag>.prettify()

    ## search methods
    <>.find_all(name, attrs, recursive, string, **kwargs)
    * name : 对标签名称的检索字符串， 可以使任一类型的过滤器,字符串,正则表达式,列表,方法或是 True . True表示返回所有。
    * attrs: 对标签属性值的检索字符串，可标注属性检索
    * recursive: 是否对子孙全部检索，默认True
    * string: <>…</>中字符串区域的检索字符串
        * <tag>(..) 等价于 <tag>.find_all(..)
        * soup(..) 等价于 soup.find_all(..)
    
    ## get rid of symbols of the html, just choose the text
    <tag>.get_text()
    ```

## Regular Expression, re

the tutorial can be find at: https://blog.51cto.com/u_15467780/4853227 

basic steps of re are as following:
* 将正则表达式的字符串形式编译为 pattern 实例；
* 使用 pattern 实例处理文本并获得一个匹配实例；
* 使用 match 实例获得所需信息。


### example
```python
import re

### findall
findall(String[, pos[, endpos]]) | re.findall(pattern, string[, flags])


### re 包括3个常见值
re.I(re.IGNORECASE)  # 使匹配忽略大小写
re.M(re.MULTILINE)   # 允许多行匹配
re.S(re.DOTALL)      # 匹配包括换行在内的所有字符


## compile, 根据包含正则表达式的字符串创建模式对象，返回一个 pattern 对象
re.compile(pattern[, flags])


## example
string = 'A1.45, b5, 6.45, 8.82'
regex = re.compile(r"\d+\.?\d*")
print(regex.findall(string))



## match
match(string[, pos[, endpos]]) | re.match(patter, string[, flags])

## search
search(string[, pos[, endpos]]) | re.search(pattern, string[, flags])


## urllib function
import urllib.request

content = urllib.request.urlopen(url) ## get the content of the webpage

```

## Scrapy

Turtorial for this tookit can be found at: https://cloud.tencent.com/developer/article/1856557 and https://www.runoob.com/w3cnote/scrapy-detail.html


### flowchat of this tookit

<center class="half">
<img src=./Pictures/craw_note/figure1.jpeg width = 80%>
</center>

Scrapy主要包括了以下组件：

* 引擎(Scrapy Engine)
* Item 项目
* 调度器(Scheduler)
* 下载器(Downloader)
* 爬虫(Spiders)
* 项目管道(Pipeline)
* 下载器中间件(Downloader Middlewares)
* 爬虫中间件(Spider Middlewares)
* 调度中间件(Scheduler Middewares)


### basic steps

* 选择目标网站
* 定义要抓取的数据（通过Scrapy Items来完成的）
* 编写提取数据的spider
* 执行spider，获取数据
* 数据存储


### install

```python
## the following steps can be realised in the terminal
## first step 
pip install scrapy
scrapy bench  # test whether it is succeed

## second step, create the spider project
scrapy startproject ProjectName   
cd ProjectName   # direct to the object file
scrapy genspider name allow_domains  ## generate a spider

    # name = "" ：这个爬虫的识别名称，必须是唯一的，在不同的爬虫必须定义不同的名字。

    # allow_domains = [] 是搜索的域名范围，也就是爬虫的约束区域，规定爬虫只爬取这个域名下的网页，不存在的URL会被忽略。

    # start_urls = () ：爬取的URL元祖/列表。爬虫从这里开始抓取数据，所以，第一次下载的数据将会从这些urls开始。其他子URL将会从这些起始URL中继承性生成。

    # parse(self, response) ：解析的方法，每个初始URL完成下载后将被调用，调用的时候传入从每一个URL传回的Response对象来作为唯一参数，主要作用如下：


Example
scrapy genspider demo "demo.cn"


## third step, extract the data
完善spider 使用xpath等
xpath 可以从浏览器选中代码块，右键copy

## forth step, save the data
pipeline中保存数据

## fifth step, run in the terminal
scrapy crawl demo     # demo爬虫的名字

## fifth step, or run in the pycharm
from scrapy import cmdline
cmdline.execute("scrapy crawl demo".split())

```

### Directory file description


<center class="half">
<img src=./Pictures/craw_note/figure2.png width = 80%>
</center>


* the function of each file:
 
    scrapy.cfg ：项目的配置文件

    mySpider/ ：项目的Python模块，将会从这里引用代码

    mySpider/items.py ：项目的目标文件

    mySpider/pipelines.py ：项目的管道文件

        * 验证爬取的数据(检查item包含某些字段，比如说name字段)
        * 查重(并丢弃)
        * 将爬取结果保存到文件或者数据库中
    mySpider/settings.py ：项目的设置文件

        * most of the time, set the parameter ROBOTSTXT_OBEY = False
    mySpider/spiders/ ：存储爬虫代码目录

        * 使用scrapy genspider demo之后会生成一个demo.py文件
        * 该文件内部有会有DemoSpider的类，里面包含了allowed_domain, start_urls等参数，这些都可以根据实际任务进行修改！
        * 此外，在该大类下，会定义解码器，parse(self, response)，所有的数据操作任务都可定义在该函数下面。response为解码后返回来的页面文本。
        * 页面文本在respond.body之下，但是response.body类型为byte，需要用‘utf-8’解码成string。

### Selectors
Scrapy Selectors 内置 XPath 和 CSS Selector 表达式机制


Selector有四个基本的方法，最常用的还是xpath:

* xpath(): 传入xpath表达式，返回该表达式所对应的所有节点的selector list列表
* extract(): 序列化该节点为字符串并返回list
* css(): 传入CSS表达式，返回该表达式所对应的所有节点的selector list列表，语法同 BeautifulSoup4
* re(): 根据传入的正则表达式对数据进行提取，返回字符串list列表


### ***example can be found at the github file***





