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

Turtorial for this tookit can be found at: https://cloud.tencent.com/developer/article/1856557 


### 

### install

```python
## the following steps can be realised in the terminal
## first step 
pip install scrapy

## second step, test whether it is succeed
scrapy bench

## third step, create the spider file
scrapy startproject ProjectName   
cd ProjectName   # direct to the object file
scrapy genspider 

```

